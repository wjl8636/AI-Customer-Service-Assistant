from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.config import settings
from app.core import budget


def count_tokens(messages) -> int:
    """估算这批消息的 token。

    count_tokens_approximately 默认 chars_per_token=4.0 是按英文定的,中文一个字往往
    就是一个 token,照默认口径算会低估四倍(实测 29 个汉字被估成 12)。裁剪、预算判断
    都走这一个入口,口径才对得齐。"""
    return count_tokens_approximately(
        messages, chars_per_token=settings.zh_chars_per_token)


def chars_to_tokens(n_chars: int) -> int:
    """按字数估 token,口径与 count_tokens 一致。

    zh_chars_per_token 是「几个字折一个 token」,所以是除不是乘。这两个方向差 1.44 倍,
    而且错的那一头没有任何报错——只是判据悄悄比预算严一截。凡是拿字数换 token 的地方
    都走这里,别再各写各的。"""
    return int(n_chars / settings.zh_chars_per_token)


def tokens_to_chars(n_tokens: int) -> int:
    """反过来:按 token 估字数,给用户看的提示语用。跟 chars_to_tokens 是一对,别写反。"""
    return int(n_tokens * settings.zh_chars_per_token)


def window_budget() -> int:
    """滑窗能用多少 token。配置里 context_window_max_tokens 填了非 0 就以它为准
    (回滚用),否则按窗口预算算。"""
    if settings.context_window_max_tokens > 0:
        return settings.context_window_max_tokens
    return budget.compute().sliding


class SessionStore:
    """内存会话存储,session_id -> 消息列表。本章不做持久化。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )


# ---- ch07 锚点切窗 + 摘要注入(三层上下文的滑窗层)----

_DB_ID_PREFIX = "db-"


def _db_msg_id(m: BaseMessage) -> int | None:
    """从消息 id 解出 MySQL 消息 id(入口约定 HumanMessage.id=f"db-{msg_id}");非锚点返回 None。"""
    mid = getattr(m, "id", None)
    if isinstance(mid, str) and mid.startswith(_DB_ID_PREFIX):
        try:
            return int(mid[len(_DB_ID_PREFIX):])
        except ValueError:
            return None
    return None


def _index_after(messages: list[BaseMessage], msg_id: int) -> int:
    """第一条 db-id 大于 msg_id 的用户消息在列表里的下标;找不到返回 0。"""
    if not msg_id:
        return 0
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            did = _db_msg_id(m)
            if did is not None and did > msg_id:
                return i
    return 0


def build_window(messages: list[BaseMessage], summary_upto_msg_id: int,
                 layer1_from_msg_id: int = 0,
                 max_tokens: int | None = None) -> list[BaseMessage]:
    """按两个锚点把历史渲染成三段:

      id ≤ summary_upto_msg_id            已进摘要,不出现在窗口里
      summary_upto < id ≤ layer1_from     层 2,渲染成半压形态
      id > layer1_from                    层 1,原样

    锚点缺失(旧会话)或为 0 时那一刀不切,行为退化成单层原文 + token 兜底裁剪。
    消息本身不动,变的只是渲染方式,所以同一份输入渲染多少次结果都一样。
    只在调模型前现拼,不改 State。"""
    window = messages[_index_after(messages, summary_upto_msg_id):]
    if layer1_from_msg_id:
        split = _index_after(window, layer1_from_msg_id)
        if split:
            window = to_layer2(window[:split]) + window[split:]
    trimmed = trim_history(window, max_tokens=max_tokens or window_budget())
    return trimmed or window


def layer_tokens(messages: list[BaseMessage], summary_upto_msg_id: int,
                 layer1_from_msg_id: int) -> tuple[int, int]:
    """量出层 2、层 1 各占多少 token,轮末据此判断要不要降级。"""
    window = messages[_index_after(messages, summary_upto_msg_id):]
    split = _index_after(window, layer1_from_msg_id) if layer1_from_msg_id else 0
    return count_tokens(to_layer2(window[:split])), count_tokens(window[split:])


def next_layer1_from(messages: list[BaseMessage], summary_upto_msg_id: int,
                     layer1_budget: int) -> int:
    """层 1 超预算时,算新的层 1 起点:从最新往回累加,累到预算为止。

    一次挪到位,不是每轮微调——两次降级之间边界不动,渲染才逐字节稳定,
    prompt cache 的前缀才保得住。"""
    window = messages[_index_after(messages, summary_upto_msg_id):]
    total = 0
    for i in range(len(window) - 1, -1, -1):
        total += count_tokens([window[i]])
        if total > layer1_budget:
            for j in range(i, -1, -1):       # 边界落在这条之前最近的一个用户消息锚点上
                did = _db_msg_id(window[j])
                if did is not None:
                    return did
            return 0
    return 0


def summary_line(summary: str | None) -> str:
    """coref/意图的历史文本前缀:单行摘要;无摘要空串。"""
    return f"(早前对话摘要:{summary})" if summary else ""


def summary_system(summary: str | None) -> SystemMessage | None:
    """把摘要包成带标题的文本块;无摘要 None。

    返回的是 SystemMessage,但调用点只取它的 .content,并进本轮材料那条 HumanMessage
    (见 nodes._turn_context)。整条消息列表里只允许有一条 SystemMessage,摘要单独占一条
    会把工具定义挤到可变内容之后,前缀缓存整段作废 —— 实测命中直接归零。"""
    if not summary:
        return None
    return SystemMessage(f"## 早前对话摘要(更早轮次已压缩,其中事实可信)\n{summary}")


# ---- 层 2:半压形态 ----
# 实测一段历史里用户消息只占 2%,客服答复 43%,工具结果 54%。所以压后两者、
# 留用户原话:丢的是自己说过的话,留下的是「那双跑鞋」「订单 1001」这些指代线索。

def compress_reply(text: str, keep_chars: int | None = None) -> str:
    """客服答复降到层 2 的形态:留头部。客服答复第一句通常就是结论。"""
    n = settings.layer2_reply_keep_chars if keep_chars is None else keep_chars
    if not text or len(text) <= n:
        return text
    return text[:n] + "…(略)"


def compress_tool_result(name: str, content: str) -> str:
    """工具结果降到层 2 的形态。

    小结果(订单字段那种)原样留着,下一轮追问还用得上;大块的(检索证据 3500 字)
    压成一行标识——那份证据是某一轮用来答某一个问题的,答完就不该再占位,
    真要数据下次实时查,比留着的旧值还新。"""
    if not content:
        return content
    if chars_to_tokens(len(content)) <= settings.layer2_tool_keep_tokens:
        return content
    return f"(已调用 {name or '工具'},结果从略)"


def to_layer2(messages: list[BaseMessage]) -> list[BaseMessage]:
    """把一批消息渲染成层 2 形态。纯函数:同一批消息进来,结果恒定,重算多少次都一样。"""
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(m)                      # 用户原话一字不动
        elif isinstance(m, AIMessage):
            # tool_calls 必须原样带上:后面那条 ToolMessage 拿 tool_call_id 指着它,
            # 这里新建一条消息时漏掉,上游会报「tool result's tool id not found」400。
            # 压的是正文,不是这一轮的调用结构。
            text = m.content if isinstance(m.content, str) else ""
            out.append(AIMessage(compress_reply(text), id=m.id,
                                 tool_calls=list(m.tool_calls or [])))
        elif isinstance(m, ToolMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)
            out.append(ToolMessage(
                compress_tool_result(getattr(m, "name", ""), text),
                tool_call_id=m.tool_call_id, name=getattr(m, "name", None), id=m.id))
        else:
            out.append(m)
    return out
