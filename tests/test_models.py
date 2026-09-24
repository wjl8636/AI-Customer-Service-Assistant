from app.db.models import Conversation, Message


async def test_conversation_defaults_and_autoincrement(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        await s.refresh(conv)               # 读回 DB 生成的 server_default(status/created_at)
        assert conv.id is not None          # BIGINT 自增
        assert conv.status == "进行中"        # DB DEFAULT 读回
        assert conv.created_at is not None


async def test_message_json_tool_calls_roundtrip(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.flush()
        msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=None,
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
        )
        s.add(msg)
        await s.commit()
        got = await s.get(Message, msg.id)
        assert got.tool_calls[0]["name"] == "query_order"   # JSON 往返
        assert got.role == "assistant"
