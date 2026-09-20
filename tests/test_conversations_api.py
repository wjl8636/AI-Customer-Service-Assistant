# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch07 验收配套:会话列表 / 历史消息两个只读接口(路由层,repository 打桩;
真库行为在 test_repository.py::test_list_conversations_* 覆盖)。"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def conv_client():
    from app.api.conversations import router
    test_app = FastAPI()
    test_app.include_router(router)
    return TestClient(test_app)


def test_list_conversations_passes_user_id(conv_client, monkeypatch):
    from app.api import conversations as api
    seen = {}

    async def fake_list(user_id):
        seen["user_id"] = user_id
        return [{"id": 2, "status": "进行中", "preview": "猫粮怎么选",
                 "has_summary": False, "updated_at": None},
                {"id": 1, "status": "进行中", "preview": "订单1001到哪了",
                 "has_summary": True, "updated_at": None}]
    monkeypatch.setattr(api.repository, "list_conversations", fake_list)

    r = conv_client.get("/api/conversations", params={"user_id": "u1"})
    assert r.status_code == 200 and seen["user_id"] == "u1"
    items = r.json()["items"]
    assert [it["id"] for it in items] == [2, 1]
    assert items[1]["has_summary"] is True


def test_list_messages_serializes_dialog(conv_client, monkeypatch):
    from app.api import conversations as api

    async def fake_get(cid):
        return object()

    async def fake_msgs(cid):
        return [SimpleNamespace(role="user", content="订单1001到哪了", created_at=None),
                SimpleNamespace(role="assistant", content="在路上", created_at=None)]
    monkeypatch.setattr(api.repository, "get_conversation", fake_get)
    monkeypatch.setattr(api.repository, "list_dialog_messages", fake_msgs)

    r = conv_client.get("/api/conversations/7/messages")
    assert r.status_code == 200
    assert [(m["role"], m["content"]) for m in r.json()["items"]] == [
        ("user", "订单1001到哪了"), ("assistant", "在路上")]


def test_list_messages_404_when_missing(conv_client, monkeypatch):
    from app.api import conversations as api

    async def fake_get(cid):
        return None
    monkeypatch.setattr(api.repository, "get_conversation", fake_get)
    assert conv_client.get("/api/conversations/999999/messages").status_code == 404
