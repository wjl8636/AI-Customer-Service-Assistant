# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from app.db import repository as repo
from app.db.models import Conversation, Ticket


async def test_create_and_get_conversation(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv is not None and conv.user_id == "u1"
    assert await repo.get_conversation(999999) is None


async def test_append_and_list_messages_in_order(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="订单 1001 到哪了")
    await repo.append_message(cid, "assistant", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    await repo.append_message(cid, "tool", content='{"status":"运输中"}', tool_call_id="c1")
    msgs = await repo.list_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[2].tool_call_id == "c1"


async def test_create_ticket_writes_and_flips_conversation_status(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    no = await repo.create_ticket(cid, "要退货", "售后")
    assert no.startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, no)
        assert t.ticket_type == "售后" and t.status == "待处理"
        conv = await s.get(Conversation, cid)
        assert conv.status == "已转人工"          # 会话状态流转


# ---- ch07 会话上下文管理 ----


async def test_conversation_summary_roundtrip(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv.summary is None and conv.summary_upto_msg_id is None   # 新会话两字段为空
    await repo.update_conversation_summary(cid, "用户问过订单1001的物流", 5)
    conv = await repo.get_conversation(cid)
    assert conv.summary == "用户问过订单1001的物流"
    assert conv.summary_upto_msg_id == 5


async def test_count_messages_after(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    ids = [await repo.append_message(cid, "user", content=f"q{i}") for i in range(4)]
    assert await repo.count_messages_after(cid, None) == 4      # 边界空=数全量
    assert await repo.count_messages_after(cid, ids[1]) == 2
    assert await repo.count_messages_after(cid, ids[3]) == 0


async def test_list_dialog_messages_filters_tool_rows(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="订单1001到哪了")
    await repo.append_message(cid, "tool", content='{"s":1}', tool_call_id="c1")
    await repo.append_message(cid, "assistant", content="在路上")
    msgs = await repo.list_dialog_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant"]      # tool 行过滤,id 升序


async def test_list_conversations_desc_preview_and_summary_flag(db_session_factory, db_clean):
    c1 = await repo.create_conversation("u1")
    await repo.append_message(c1, "user", content="订单1001到哪了,我着急要")
    c2 = await repo.create_conversation("u1")
    await repo.create_conversation("someone-else")     # 别人的会话不可见
    await repo.update_conversation_summary(c1, "用户问过订单1001", 1)

    items = await repo.list_conversations("u1")
    assert [it["id"] for it in items] == [c2, c1]      # 新会话在前
    assert items[1]["preview"].startswith("订单1001")
    assert items[0]["preview"] == "(空会话)"
    assert [it["has_summary"] for it in items] == [False, True]
