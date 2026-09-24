"""ch09 可观测:Langfuse 挂载与 trace 标注,全部可选降级。

课程 README 姿势:设三个环境变量 + 图编译时挂一次回调,节点业务代码零侵入。
项目配置经 pydantic-settings(.env),不依赖 os.environ,故显式传参初始化单例。
所有对外函数在未配置 Langfuse 时必须是安全 no-op——观测是增强,不是依赖。

实装 SDK 为 langfuse 4.x:构造参 base_url(host 已弃用);给运行中的 trace 补
metadata/tags 用 propagate_attributes 上下文管理器(v3 的 update_current_trace 已移除)。
"""
import logging

from app.config import settings

logger = logging.getLogger(__name__)
_client = None


def langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key
                and settings.langfuse_base_url)


def _init_client():
    """初始化(或复用)Langfuse 单例。"""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
    return _client


def _make_handler():
    from langfuse.langchain import CallbackHandler

    class _InlineCallbackHandler(CallbackHandler):
        """run_inline=True:让 LangChain 在调用方协程内联执行回调,而不是丢进独立 task。
        没有它,handler 的 OTel context.attach 挂在别的 task 上,节点代码里没有活动
        span 上下文,tag_intent 的 propagate_attributes 写不进当前 trace(实测踩坑)。"""
        run_inline = True

    return _InlineCallbackHandler()


def get_langfuse():
    """启用时返回已初始化 client(cost 脚本用 client.api 查询);未启用返回 None。"""
    if not langfuse_enabled():
        return None
    return _init_client()


def attach_observability(graph):
    """编译后挂一次 Langfuse 回调,全图自动 trace(README:「编译时挂一次,节点里一行不用动」)。
    未配置时原样返回——不挂、不 import langfuse。"""
    if not langfuse_enabled():
        return graph
    _init_client()
    return graph.with_config({"callbacks": [_make_handler()]})


def tag_intent(intent: str, confidence: float) -> None:
    """把意图写进当前 trace 的 metadata + tag(Cost Control 按意图分堆的钩子)。
    任何异常静默降级——观测失败不许影响业务流。"""
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client, propagate_attributes
        # 不能只 with...pass 写"当前活动 span"——LangGraph 异步调度下它可能已 ended,
        # OTel 对 ended span 的写入静默丢弃(实测踩坑)。改为在传播上下文内新建一个
        # event 观测:新 span 必然存活,trace 级属性(metadata/tags)随它上提到整条 trace。
        with propagate_attributes(
            metadata={"intent": intent, "intent_confidence": f"{confidence:.2f}"},
            tags=[f"intent:{intent}"],
        ):
            get_client().create_event(name="intent-tagged")
    except Exception:
        logger.warning("langfuse tag_intent 失败(已忽略)", exc_info=True)
