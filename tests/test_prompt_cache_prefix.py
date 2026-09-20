# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""前缀缓存不变量:发给模型的消息里,可变内容一律不许进 SystemMessage。

各家 prompt caching 都按前缀精确匹配。system 是 messages[0],把每轮都变的证据/订单数据
拼进去,等于让 system + tools 这段稳定前缀(本项目约 1664 tokens)每次请求全部 miss,
而 ReAct 环每一步都重发一次 system,损失是乘上去的。

这组测试把「system 恒定」和「插入位置」两件事钉死——注释会被无视,单测不会。
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.prompts import AGENT_SYSTEM
from app.graph.nodes import _agent_messages

EVIDENCE = "[1] 退货政策: 7天无理由"
ORDER = {"order_id": "1001", "status": "已签收", "product": "猫粮 5kg"}

STATES = [
    ("business", {"route": "business", "messages": [HumanMessage("订单1001到哪了")]}),
    ("knowledge", {"route": "knowledge", "evidence": EVIDENCE,
                   "messages": [HumanMessage("能退吗")]}),
    ("refund", {"route": "refund_flow", "evidence": EVIDENCE, "order_data": ORDER,
                "messages": [HumanMessage("这单能退吗")]}),
    ("knowledge+摘要", {"route": "knowledge", "evidence": EVIDENCE,
                        "summary": "用户问过订单1001", "messages": [HumanMessage("能退吗")]}),
]


@pytest.mark.parametrize("name,state", STATES, ids=[s[0] for s in STATES])
def test_system_message_is_always_the_static_prompt(name, state):
    """out[0] 恒等于 AGENT_SYSTEM。精确相等,不是 in——这是防「又把可变内容拼回去」的关键断言。"""
    msgs = _agent_messages(state)
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == AGENT_SYSTEM


def test_system_prefix_identical_across_routes():
    """不同路由、带不带证据,system 都得逐字一样,前缀才复用得上。"""
    heads = {_agent_messages(s)[0].content for _, s in STATES}
    assert len(heads) == 1


@pytest.mark.parametrize("name,state", STATES, ids=[s[0] for s in STATES])
def test_exactly_one_system_message(name, state):
    """整条列表只许有一条 SystemMessage。

    上游 chat template 会把所有 system 上提合并成一个头部块渲染,多出来的任何一条
    (比如曾经单独占位的摘要)都会拼在 AGENT_SYSTEM 后面,把工具 schema 挤到可变内容
    之后。实测:摘要一进 system,cache_read 从 2048 掉到 0。
    """
    assert sum(isinstance(m, SystemMessage) for m in _agent_messages(state)) == 1


@pytest.mark.parametrize("name,state", STATES, ids=[s[0] for s in STATES])
def test_volatile_payload_never_lands_in_any_system_message(name, state):
    """摘要那条 system 也不许混进可变载荷。"""
    systems = " ".join(m.content for m in _agent_messages(state) if isinstance(m, SystemMessage))
    for volatile in ("[1] 退货政策", "猫粮 5kg", "已检索到的知识证据", "退款判定任务",
                     "用户问过订单1001"):
        assert volatile not in systems


def test_step1_prompt_is_strict_prefix_of_step2():
    """ReAct 第 1 步的整段 prompt 必须是第 2 步的严格前缀,轮内才命中。

    这条同时锁死了插入位置:谁把本轮材料改成追加到列表末尾、或塞到滑窗前面,这里立刻红。
    """
    base = {"route": "refund_flow", "evidence": EVIDENCE, "order_data": ORDER}
    q = HumanMessage("这单能退吗", id="db-1")
    step1 = _agent_messages({**base, "messages": [q]})
    step2 = _agent_messages({**base, "messages": [
        q,
        AIMessage("", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"},
                                   "id": "call_1"}]),
        ToolMessage("已签收", tool_call_id="call_1", name="query_order"),
    ]})
    shape = lambda ms: [(m.type, m.content) for m in ms]      # noqa: E731
    assert shape(step2)[:len(step1)] == shape(step1)


def test_turn_context_is_not_written_back_to_state():
    """本轮材料只在拼装时现拼,不许落进 State——落了就会污染跨轮历史与审计。"""
    msgs_in = [HumanMessage("能退吗")]
    state = {"route": "knowledge", "evidence": EVIDENCE, "messages": msgs_in}
    _agent_messages(state)
    assert state["messages"] is msgs_in and len(msgs_in) == 1
