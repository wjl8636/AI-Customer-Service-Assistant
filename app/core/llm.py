# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import logging

from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from app.config import settings

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def _diff_usage(cur: dict, prev: dict) -> dict:
    """把累计用量换成相对上一个 chunk 的增量,嵌套的 *_details 一起换。"""
    out: dict = {}
    for k, v in cur.items():
        p = prev.get(k)
        if isinstance(v, dict):
            out[k] = _diff_usage(v, p if isinstance(p, dict) else {})
        elif isinstance(v, (int, float)):
            out[k] = max(0, v - (p if isinstance(p, (int, float)) else 0))
        else:
            out[k] = v
    return out


class _CumulativeUsageChatOpenAI(ChatOpenAI):
    """修正流式用量被重复累加的问题。

    有些 OpenAI 兼容通道在**每个** chunk 上都重报一遍**累计**用量,而 langchain-openai
    把各 chunk 的用量**相加**。两件事凑到一起,一次调用的 token 数会被放大到 chunk 条数倍
    ——实测 24 个 chunk、input 恒为 11,聚合出来是 264。ch09 的成本账按这个数统计,
    整体虚高一个量级;ch05 早先那道 token 预算闸也因此第一步就跳闸,工具从没执行过。
    开关救不了:`stream_usage=False` 时上游照发、库照加。

    修法是在 chunk 转换的入口把累计值换成增量,再交给上游逻辑去加,加完正好等于最后一个
    chunk 的累计值。只在末尾报一次用量的规范通道不受影响(增量就等于它本身)。
    """

    _prev_usage: dict = PrivateAttr(default_factory=dict)

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        usage = chunk.get("usage") if isinstance(chunk, dict) else None
        if usage:
            prev = self._prev_usage
            # 任何一项变小 = 上一条流已经结束,这是新一条流的第一个 chunk,基准清零。
            # 同一实例被复用时(结构化输出的链会多次 invoke)靠这条自愈。
            if any(isinstance(v, (int, float)) and v < prev.get(k, 0)
                   for k, v in usage.items() if not isinstance(v, dict)):
                prev = {}
            chunk = {**chunk, "usage": _diff_usage(usage, prev)}
            self._prev_usage = usage
        return super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info)


def resolve_slot(slot: str = "chat") -> tuple[str, str, str]:
    """按用途取 (模型名, 地址, key)。intent / summary 各是一个槽位,三项分别留空就
    回落 CHAT 那组——只用一家上游时它们一个都不用填,想把某个用途换到别家才填。"""
    if slot == "chat":
        return settings.chat_model, settings.chat_base_url, settings.chat_api_key
    return (
        getattr(settings, f"{slot}_model", "") or settings.chat_model,
        getattr(settings, f"{slot}_base_url", "") or settings.chat_base_url,
        getattr(settings, f"{slot}_api_key", "") or settings.chat_api_key,
    )


def get_chat_model(streaming: bool = False, model: str | None = None,
                   temperature: float | None = None, slot: str = "chat",
                   thinking: bool = True) -> ChatOpenAI:
    """直连聊天上游,说 OpenAI 协议。
    model 显式覆盖模型名(意图识别可配更大模型求准),None 时回落 settings.chat_model。
    ch09:usage 统一收口——流式也回传 token(等效 stream_options.include_usage),
    否则 Langfuse 按意图统计只有非流式调用的账,主力 Agent 全漏。

    temperature 默认 0.3(答用户的话要有点人味)。**当评委用的链路要显式传 0**:
    同一份证据+同一份答案,0.3 下重放四次给出过 5/6、5/6、6/6、6/6 三种结果,
    量出来的分数里就混进了采样噪声——评估要的是可复现,不是多样性。

    thinking=False 显式关闭思考链。structured() 用它:DeepSeek 的 thinking 和
    tool_choice 指定函数互斥(官方文档:思考模式下不支持 required 和指定具体 tool),
    结构化输出走 function_calling 必须关。不能只是不传——DeepSeek 默认 thinking 开着,
    不传等于开,所以 False 时显式发 disabled。"""
    slot_model, base_url, api_key = resolve_slot(slot)
    thinking_kw = _thinking_kwargs() if thinking else {"extra_body": {"thinking": {"type": "disabled"}}}
    return _CumulativeUsageChatOpenAI(
        model=model or slot_model,
        base_url=base_url,
        api_key=api_key,
        streaming=streaming,
        stream_usage=True,
        max_tokens=settings.max_output_tokens,   # Y:保险丝,防模型失控输出
        temperature=0.3 if temperature is None else temperature,
        **thinking_kw,
    )


def _thinking_kwargs() -> dict:
    """把三项思考链配置拼成 ChatOpenAI 的参数。留空的不传,交给上游默认。

    thinking 和 reasoning_split 不是 OpenAI 协议里的标准字段,得走 extra_body;
    reasoning_effort 是标准字段,ChatOpenAI 直接认。上游不认的参数会被静默忽略
    (给不认 thinking.type 的上游传过去,既不报错也不生效),所以这里不按 provider 分叉。"""
    kwargs: dict = {}
    extra: dict = {}
    if settings.chat_thinking:
        extra["thinking"] = {"type": settings.chat_thinking}
    if settings.chat_reasoning_split:
        # 只认字面量 "true" 会把填 1 / yes / on 的人坑了:那几种写法本意是开,却会被转成
        # False 显式发出去,等于亲手关掉这项兜底。填 false / 0 才是真的要关。
        extra["reasoning_split"] = settings.chat_reasoning_split.strip().lower() in _TRUTHY
    if extra:
        kwargs["extra_body"] = extra
    if settings.chat_reasoning_effort:
        kwargs["reasoning_effort"] = settings.chat_reasoning_effort
    return kwargs


# ---------- 结构化输出 ----------
# 只走 function calling 一条路,开流式,结果只认 tool_calls。三条都不是随手选的:
#
# 中转的 OpenAI 兼容层在**非流式**下会把 arguments 拼坏,前面粘一个空对象变成
# `{}{"faithful":...}`。同一个中转、同一个模型换成流式,arguments 就是干净的合法
# JSON,所以这个坑靠开流式绕过。
#
# 机制也必须写死成 function_calling:langchain-openai 的 ChatOpenAI 默认是
# json_schema,那条路解析的是正文,而模型有时会在 tool_calls 旁边额外吐一段思考正文,
# 解析正文就会拿到自然语言、返回 None。只认 tool_calls 则不受这个随机形态影响。
#
# 不留 json_mode 这条降级路。json_mode 自己不带 schema,字段定义只能塞进系统消息里
# 靠模型自觉,本来就比 function calling 弱一截,个别上游还会把 JSON Schema 原样吐回来、
# 解析必炸。降级到一条会炸的路,不如当场失败:失败会抛到调用点,那里降级成安全默认值
# (检索用原问题、自评判证据不足),系统照样答得出话,日志里还留得下真正的出错位置。
#
# 「失败」有两种形态,漏掉哪种都不行:抛异常(arguments 被拼坏),以及**返回 None 不抛**
# ——响应里没有 tool_calls 时 PydanticToolsParser(first_tool_only=True) 就是给 None。
# 只认异常的话,None 会一路传到调用点上炸(intent 的 r.intent、selfcheck 的 r.useful),
# 报错位置还指不到真正的出处,所以下面把 None 就地转成异常。

# 结构化输出必须走非流式的模型族。两类上游的毛病正好相反:中转的非流式 OpenAI 兼容层会把
# tool_calls 的 arguments 拼坏(所以 structured 默认开流式),下面这几个族反过来,流式下
# 压根不吐 tool_calls,非流式才给。换模型踩到同样的坑,往这个元组里加一项。
_NO_STREAM_TOOLCALL_FAMILIES: tuple[str, ...] = ("minimax",)


def needs_non_streaming_tools(model: str | None = None, slot: str = "chat") -> bool:
    name = (model or resolve_slot(slot)[0] or "").lower()
    return any(f in name for f in _NO_STREAM_TOOLCALL_FAMILIES)


def structured(schema, *, model: str | None = None, streaming: bool = True,
               temperature: float | None = None, slot: str = "chat"):
    """要结构化结果就用这个,别直接 with_structured_output——它默认走 json_schema,
    那条路解析正文,会被模型多吐的思考正文带崩。

    默认开流式:中转的非流式 OpenAI 兼容层会把 tool_calls 的 arguments 拼坏,流式不会
    (见上面那段注释里的实测)。

    只有 function calling 一条通路。偶发失败原样重试一次,再不成就把异常抛给调用点——
    调用点各自降级成安全默认值,别在这里降级到一条更差的通路上。

    temperature 透传给底层模型:当评委用的结构化链路传 0(见 get_chat_model 的说明)。
    """
    from langchain_core.runnables import RunnableLambda

    slot_model, slot_base, _ = resolve_slot(slot)
    name = (model or slot_model or "").lower()
    if needs_non_streaming_tools(model, slot=slot):
        streaming = False        # 这类模型流式不吐 tool_calls,见 _NO_STREAM_TOOLCALL_FAMILIES
    m = get_chat_model(streaming=streaming, model=model, temperature=temperature,
                       slot=slot, thinking=False)
    fc = m.with_structured_output(schema, method="function_calling")

    def _log(attempt: int, why: str) -> None:
        # 带上通道:同一个模型换个上游行为就不一样,只打模型名排障时定位不到
        logger.warning("%s 经 %s 的 function calling 第 %s 次没成(%s),%s",
                       name, slot_base, attempt, why,
                       "原样重试一次" if attempt == 1 else "抛给调用点")

    def _no_tool_calls() -> OutputParserException:
        return OutputParserException(
            f"{name} 经 {settings.chat_base_url} 连着两次都没返回 tool_calls")

    def _run(x):
        for attempt in (1, 2):     # 偶发不返回 tool_calls,原样重试一次
            try:
                r = fc.invoke(x)
            except Exception as e:
                _log(attempt, type(e).__name__)
                if attempt == 2:
                    raise
                continue
            if r is not None:
                return r
            _log(attempt, "响应里没有 tool_calls")
        raise _no_tool_calls()

    async def _arun(x):
        for attempt in (1, 2):
            try:
                r = await fc.ainvoke(x)
            except Exception as e:
                _log(attempt, type(e).__name__)
                if attempt == 2:
                    raise
                continue
            if r is not None:
                return r
            _log(attempt, "响应里没有 tool_calls")
        raise _no_tool_calls()

    return RunnableLambda(_run, afunc=_arun)
