from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.graph.state import ConversationState, merge_dict


def test_merge_dict_accumulates():
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dict(None, {"b": 2}) == {"b": 2}
    assert merge_dict({"a": 1}, {"a": 9}) == {"a": 9}
    assert merge_dict({"a": 1}, None) == {}          # None=重置哨兵:入口每轮清零 trace,杜绝跨轮泄漏


def test_add_messages_reducer_appends():
    merged = add_messages([HumanMessage("hi")], [AIMessage("yo")])
    assert [m.content for m in merged] == ["hi", "yo"]


def test_state_has_required_keys():
    keys = ConversationState.__annotations__
    for k in ["messages", "user_id", "conversation_id", "intent", "route",
              "evidence", "citations", "evidence_strong", "answer",
              "steps", "tokens_used", "suggested_actions", "trace"]:
        assert k in keys
