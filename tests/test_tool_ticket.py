from app.db.models import Conversation, Ticket
from app.tools.builtin.tickets import create_ticket


def test_conversation_id_hidden_from_model_schema():
    # InjectedToolArg 的参数不应出现在暴露给 LLM 的 args schema 里。
    #
    # 注意:get_input_schema() 返回的是「完整」输入 schema(含 InjectedToolArg 参数,
    # 因为 ainvoke 时仍需要这个值),不会做注入参数过滤——实测(langchain-core 1.4.9)
    # 其 properties 里 description/ticket_type/conversation_id 三者都在。
    # 真正「模型可见」的 schema 是 tool_call_schema(BaseTool.args 属性、以及
    # bind_tools 时 convert_to_openai_tool 用的都是它),它才会按 InjectedToolArg
    # 过滤掉注入参数——实测其 properties 只剩 description/ticket_type,
    # 与 convert_to_openai_tool(create_ticket) 生成的 OpenAI function schema 一致。
    # 故本测试断言用 tool_call_schema,才是对「模型看不到 conversation_id」的正确验证。
    schema = create_ticket.tool_call_schema.model_json_schema()
    props = schema.get("properties", {})
    assert "description" in props and "ticket_type" in props
    assert "conversation_id" not in props        # 关键:模型看不到会话主键


async def test_create_ticket_injects_conversation_id_and_writes(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id
    r = await create_ticket.ainvoke(
        {"description": "商品损坏要退货", "ticket_type": "售后", "conversation_id": cid}
    )
    assert r["ticket_no"].startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, r["ticket_no"])
        assert t is not None and t.conversation_id == cid
