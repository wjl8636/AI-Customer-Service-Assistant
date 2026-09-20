# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.exc import SQLAlchemyError

from app.api import agent
from app.graph import runtime
from app.main import app


def _fake_state_with_tool_trace():
    ai = AIMessage("", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    tm = ToolMessage(content='{"status":"运输中"}', tool_call_id="c1", name="query_logistics")
    final_ai = AIMessage("订单 1001 正在运输中。")
    return {
        "messages": [HumanMessage("订单 1001 到哪了"), ai, tm, final_ai],
        "conversation_id": 12,
    }


def test_agent_endpoint_returns_tool_trace(monkeypatch):
    async def fake_run_turn(user_id, message, conversation_id):
        return {"conversation_id": 12, "state": _fake_state_with_tool_trace()}
    monkeypatch.setattr(agent.runtime, "run_turn", fake_run_turn)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "订单 1001 到哪了"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 12
    assert body["answer"] == "订单 1001 正在运输中。"
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["ok"] is True
    assert body["suggested_actions"] == []


def test_agent_endpoint_chitchat_has_no_tool_calls(monkeypatch):
    async def fake_run_turn(user_id, message, conversation_id):
        state = {"messages": [HumanMessage("你好"), AIMessage("你好呀,有什么可以帮你?")],
                 "answer": "你好呀,有什么可以帮你?", "conversation_id": 13}
        return {"conversation_id": 13, "state": state}
    monkeypatch.setattr(agent.runtime, "run_turn", fake_run_turn)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "你好"})
    assert r.status_code == 200
    body = r.json()
    assert body["tool_calls"] == []
    assert body["tool_results"] == []


def test_agent_endpoint_dedups_suggested_actions(monkeypatch):
    async def fake_run_turn(user_id, message, conversation_id):
        state = {
            "messages": [HumanMessage("投诉"), AIMessage("已收到你的反馈。")],
            "answer": "已收到你的反馈。",
            "conversation_id": 14,
            "suggested_actions": [
                {"type": "transfer_human"},
                {"type": "create_ticket", "draft": {"n": 1}},
                {"type": "create_ticket", "draft": {"n": 2}},
            ],
        }
        return {"conversation_id": 14, "state": state}
    monkeypatch.setattr(agent.runtime, "run_turn", fake_run_turn)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "投诉"})
    assert r.status_code == 200
    body = r.json()
    assert body["suggested_actions"] == [
        {"type": "transfer_human"},
        {"type": "create_ticket", "draft": {"n": 1}},
    ]


def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run_turn(*a, **k):
        raise runtime.ConversationNotFound(999)
    monkeypatch.setattr(agent.runtime, "run_turn", fake_run_turn)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999})
    assert r.status_code == 404


def test_agent_endpoint_503_on_db_error(monkeypatch):
    async def fake_run_turn(*a, **k):
        raise SQLAlchemyError("boom")
    monkeypatch.setattr(agent.runtime, "run_turn", fake_run_turn)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 503


def test_agent_endpoint_502_on_generic_failure(monkeypatch):
    async def fake_run_turn(*a, **k):
        raise RuntimeError("upstream boom")
    monkeypatch.setattr(agent.runtime, "run_turn", fake_run_turn)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 502


def test_agent_endpoint_422_on_missing_fields():
    client = TestClient(app)
    assert client.post("/api/agent", json={"message": "hi"}).status_code == 422
