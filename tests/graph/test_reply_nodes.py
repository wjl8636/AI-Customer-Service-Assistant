import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_script_reply_fixed_no_actions():
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT
    out = await nodes.script_reply({"intent": "闲聊"})
    assert out["answer"] == SCRIPT_REPLY_CHITCHAT
    assert out["trace"]["route"] == "fallback_script"
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_complaint_reply_offers_two_actions():
    out = await nodes.complaint_reply({"messages": [HumanMessage("我要投诉你们")], "conversation_id": 7})
    assert out["answer"] == nodes.COMPLAINT_REPLY
    types = [a["type"] for a in out["suggested_actions"]]
    assert types == ["transfer_human", "create_ticket"]
    ticket = next(a for a in out["suggested_actions"] if a["type"] == "create_ticket")
    assert ticket["draft"]["ticket_type"] == "投诉"
    assert ticket["draft"]["description"] == "我要投诉你们"


@pytest.mark.asyncio
async def test_fallback_reply_records_low_confidence(monkeypatch):
    calls = {}

    async def fake_insert(conversation_id, raw_question, source, reason, retrieved_chunks=None):
        calls.update(conversation_id=conversation_id, raw=raw_question, source=source)
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    out = await nodes.fallback_reply(
        {"messages": [HumanMessage("怎么注销账号")], "conversation_id": 3,
         "trace": {"evidence_top": 0.1}}
    )
    assert out["answer"] == nodes.FALLBACK_REPLY
    assert calls["source"] == "retrieval_low_conf"
    assert calls["conversation_id"] == 3
    # 兜底话术让用户「联系人工客服」,就得把转人工按钮一并递给前端,不能光嘴上说
    assert [a["type"] for a in out["suggested_actions"]] == ["transfer_human"]
