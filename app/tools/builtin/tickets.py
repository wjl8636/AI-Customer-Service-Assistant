import asyncio
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from app.config import settings
from app.db import repository
from app.tools import registry


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """创建人工工单。仅当用户明确要求建工单/要求人工跟进时才调用;调用前必须确认 description
    (问题描述)已从用户处问清,信息不足时先向用户追问,严禁编造或用占位文本。
    ticket_type 从 售后/投诉/咨询 中选;工单关联的会话号由系统注入,你不要传。"""
    if settings.demo_ticket_delay_seconds > 0:      # 验收 6:环境变量注入写操作超时演示
        await asyncio.sleep(settings.demo_ticket_delay_seconds)
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {"ticket_no": ticket_no, "status": "已转人工"}


registry.register(registry.spec_from_langchain_tool(
    create_ticket, source="builtin", inject_conversation=True))
