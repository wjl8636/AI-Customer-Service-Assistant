"""ch09 入口3:👎 落池(source=user_feedback,快照尽力回捞);👍 只记日志不落库。"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import feedback as feedback_api


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(feedback_api.router)
    return TestClient(app)


@pytest.fixture
def spy_insert(monkeypatch):
    calls = []

    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        calls.append({"cid": cid, "q": q, "source": source, "chunks": retrieved_chunks})
        return 1
    monkeypatch.setattr(feedback_api.repository, "insert_low_confidence", fake_insert)
    return calls


def _stub_snapshot(monkeypatch, question, snapshot):
    async def fake_get(cid):
        return {"question": question, "snapshot": snapshot}
    monkeypatch.setattr(feedback_api.runtime, "get_turn_snapshot", fake_get)


def test_down_pools_with_snapshot_when_question_matches(client, spy_insert, monkeypatch):
    snap = [{"question": "x", "answer": "y", "rerank_score": 0.4, "section_path": "s"}]
    _stub_snapshot(monkeypatch, "猫窝能水洗吗", snap)
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "down", "question": "猫窝能水洗吗"})
    assert r.status_code == 200 and r.json()["pooled"] is True
    assert spy_insert[0]["source"] == "user_feedback"
    assert spy_insert[0]["chunks"] == snap        # 问题对得上 → 快照带走


def test_down_pools_without_snapshot_when_question_differs(client, spy_insert, monkeypatch):
    _stub_snapshot(monkeypatch, "另一个问题",
                   [{"question": "x", "answer": "y", "rerank_score": 0.4, "section_path": "s"}])
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "down", "question": "猫窝能水洗吗"})
    assert r.status_code == 200
    assert spy_insert[0]["chunks"] is None        # 对不上 → 空着(尽力而已,不硬塞)


def test_down_survives_snapshot_failure(client, spy_insert, monkeypatch):
    async def boom(cid):
        raise RuntimeError("checkpointer 不可用")
    monkeypatch.setattr(feedback_api.runtime, "get_turn_snapshot", boom)
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "down", "question": "猫窝能水洗吗"})
    assert r.status_code == 200                   # 回捞失败不拦落池
    assert spy_insert[0]["chunks"] is None


def test_up_logs_only(client, spy_insert, monkeypatch):
    _stub_snapshot(monkeypatch, "q", [])
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "up", "question": "q"})
    assert r.status_code == 200 and r.json()["pooled"] is False
    assert spy_insert == []                       # 👍 不落库


def test_rejects_bad_rating(client, spy_insert):
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "meh", "question": "q"})
    assert r.status_code == 422
