import pytest

from app.graph import runtime


@pytest.fixture()
def no_summary_task(monkeypatch):
    """屏蔽 ch07 轮后摘要触发(会读真库);触发逻辑本身在 test_summarizer.py 测。"""
    async def noop(cid):
        return None
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", noop)


def test_graph_input_resets_per_turn_output_channels():
    # C1 回归:无 reducer 的输出通道必须每轮清零,否则 checkpointer 会把上一轮的
    # answer/建工单可选项/证据残留到本轮(同会话跨轮泄漏)。messages 仍是追加(跨轮历史)。
    inp = runtime._graph_input("u1", "hi", 7, 1, "", 0)
    assert inp["answer"] == ""
    assert inp["suggested_actions"] == []
    assert inp["evidence"] == "" and inp["evidence_strong"] is False
    assert inp["citations"] == []
    assert inp["steps"] == 0 and inp["tokens_used"] == 0
    assert [m.content for m in inp["messages"]] == ["hi"]  # 本轮新消息(reducer 追加进历史)
    assert inp["messages"][0].id == "db-1"                 # ch07:用户消息带 DB 锚点
    assert inp["summary"] == "" and inp["summary_upto_msg_id"] == 0  # ch07:摘要两字段每轮刷新
    assert inp["resolved_query"] == "" and inp["intent_confidence"] == 0.0
    assert inp["order_id"] == "" and inp["order_data"] == {}
    # trace 是 merge-reducer 通道,塞 {} 会被合并(merge(旧,{})=旧)清不掉;
    # 传 None 作重置哨兵,经 merge_dict 清零本轮 trace(否则上一轮 retrieve_policy 等条件键残留)。
    assert inp["trace"] is None


@pytest.mark.asyncio
async def test_trace_does_not_leak_across_turns():
    # 同会话跨轮:第1轮只写一次的条件键(retrieve_policy)不得残留到第2轮。
    # 入口 trace=None → merge_dict 重置 → checkpointer 里的上轮 trace 被清。
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from app.graph.state import ConversationState

    async def writer(state):
        if state.get("resolved_query") == "turn1":
            return {"trace": {"coref": "x", "retrieve_policy": {"hits": 14}}}
        return {"trace": {"coref": "y"}}

    b = StateGraph(ConversationState)
    b.add_node("writer", writer)
    b.add_edge(START, "writer")
    b.add_edge("writer", END)
    g = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-trace-leak"}}

    out1 = await g.ainvoke({"resolved_query": "turn1", "trace": None}, cfg)
    assert out1["trace"] == {"coref": "x", "retrieve_policy": {"hits": 14}}
    out2 = await g.ainvoke({"resolved_query": "turn2", "trace": None}, cfg)
    assert out2["trace"] == {"coref": "y"}          # 上一轮 retrieve_policy 不残留
    assert "retrieve_policy" not in out2["trace"]


@pytest.mark.asyncio
async def test_run_turn_creates_conversation_when_none(monkeypatch, no_summary_task):
    async def fake_create(uid): return 42
    async def fake_append(cid, role, content=None, **k): return 1

    class FakeGraph:
        async def ainvoke(self, inp, config):
            return {"answer": "hi", "messages": [], "conversation_id": inp["conversation_id"]}

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())

    out = await runtime.run_turn("u1", "你好", None)
    assert out["conversation_id"] == 42


@pytest.mark.asyncio
async def test_run_turn_raises_when_conversation_missing(monkeypatch):
    async def fake_get(cid): return None
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    with pytest.raises(runtime.ConversationNotFound):
        await runtime.run_turn("u1", "hi", 999)


@pytest.mark.asyncio
async def test_stream_turn_maps_events(monkeypatch, no_summary_task):
    async def fake_create(uid): return 7
    async def fake_append(cid, role, content=None, **k): return 1
    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)

    from langchain_core.messages import AIMessage, ToolMessage

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            # messages 模式:main_agent 的 token
            yield ("messages", (AIMessage("已"), {"langgraph_node": "main_agent"}))
            yield ("messages", (AIMessage("发货"), {"langgraph_node": "main_agent"}))
            # messages 模式:非答复节点 token 应被过滤
            yield ("messages", (AIMessage("订单"), {"langgraph_node": "classify_intent"}))
            # updates 模式:工具帧 + citations + actions
            yield ("updates", {"agent_tools": {"messages": [
                ToolMessage(content="{}", tool_call_id="1", name="query_logistics")]}})
            yield ("updates", {"retrieve_knowledge": {"citations": [{"n": 1}]}})
            yield ("updates", {"complaint_reply": {"answer": "抱歉",
                    "suggested_actions": [{"type": "transfer_human"}]}})

    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "订单1001物流", None)]
    kinds = [e["type"] for e in events]
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    tools = [e["name"] for e in events if e["type"] == "tool"]
    assert "已" in deltas and "发货" in deltas
    assert "订单" not in deltas  # 非答复节点被过滤
    assert tools == ["query_logistics"]
    assert any(e["type"] == "citations" for e in events)
    assert any(e["type"] == "actions" for e in events)
    assert kinds[-1] == "done" and events[-1]["conversation_id"] == 7


def test_dedup_actions_keeps_first_occurrence_per_type():
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket", "payload": {"n": 1}},
        {"type": "create_ticket", "payload": {"n": 2}},
    ]
    out = runtime.dedup_actions(actions)
    assert out == [
        {"type": "transfer_human"},
        {"type": "create_ticket", "payload": {"n": 1}},
    ]


@pytest.mark.asyncio
async def test_stream_turn_dedups_actions_before_emitting(monkeypatch, no_summary_task):
    async def fake_create(uid): return 8
    async def fake_append(cid, role, content=None, **k): return 1
    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            # agent 在多个 ReAct 轮次重复提议 create_ticket
            yield ("updates", {"main_agent": {
                "suggested_actions": [{"type": "transfer_human"}, {"type": "create_ticket"}]}})
            yield ("updates", {"main_agent": {
                "suggested_actions": [{"type": "create_ticket"}]}})

    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "hi", None)]
    actions_events = [e for e in events if e["type"] == "actions"]
    assert len(actions_events) == 1
    assert actions_events[0]["items"] == [{"type": "transfer_human"}, {"type": "create_ticket"}]


# ---- Task 13: interrupt 事件 + resume ----

@pytest.mark.asyncio
async def test_stream_turn_emits_interrupt_event(monkeypatch, no_summary_task):
    async def fake_create(uid): return 9
    async def fake_append(cid, role, content=None, **k): return 1
    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)

    class Intr:
        def __init__(self, value): self.value = value

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            yield ("updates", {"fetch_order": None})
            yield ("updates", {"__interrupt__": (Intr(
                {"type": "select_order", "orders": [{"order_id": "1001"}]}),)})

    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "我要退款", None)]
    intr = [e for e in events if e["type"] == "interrupt"]
    assert intr and intr[0]["kind"] == "select_order"
    assert intr[0]["orders"] == [{"order_id": "1001"}]
    assert intr[0]["conversation_id"] == 9   # 中断带会话号供前端续跑(无 done 帧)


@pytest.mark.asyncio
async def test_resume_turn_drives_command(monkeypatch, no_summary_task):
    from langgraph.types import Command
    seen = {}

    class FakeGraph:
        async def ainvoke(self, inp, config):
            seen["is_command"] = isinstance(inp, Command)
            seen["resume"] = getattr(inp, "resume", None)
            return {"answer": "这一单可以退款", "messages": []}

    async def fake_get(cid): return object()
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    out = await runtime.resume_turn(5, "1001")
    assert seen["is_command"] and seen["resume"] == "1001"
    assert out["conversation_id"] == 5 and out["state"]["answer"] == "这一单可以退款"
