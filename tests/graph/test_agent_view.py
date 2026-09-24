from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.agent import _views_from_state


def test_views_rebuilt_from_messages():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}])
    tm = ToolMessage(content='{"status":"已发货"}', tool_call_id="c1", name="query_order")
    calls, results = _views_from_state({"messages": [HumanMessage("q"), ai, tm, AIMessage("答")]})
    assert calls[0].name == "query_order"
    assert results[0].tool_call_id == "c1"
    assert results[0].name == "query_order"


def test_views_from_state_empty_messages_returns_empty_lists():
    calls, results = _views_from_state({"messages": []})
    assert calls == []
    assert results == []


def test_views_from_state_marks_error_tool_result_not_ok():
    tm = ToolMessage(content="boom", tool_call_id="c2", name="query_order", status="error")
    calls, results = _views_from_state({"messages": [tm]})
    assert calls == []
    assert results[0].ok is False


def test_views_from_state_multiple_tool_calls_in_one_ai_message():
    ai = AIMessage("", tool_calls=[
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        {"name": "query_logistics", "args": {"order_id": "1"}, "id": "c2"},
    ])
    calls, _ = _views_from_state({"messages": [ai]})
    assert [c.name for c in calls] == ["query_order", "query_logistics"]
