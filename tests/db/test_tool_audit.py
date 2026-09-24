from sqlalchemy import select

from app.db import repository
from app.db.models import ToolAuditLog


async def test_insert_tool_audit_minimal(db_session_factory):
    await repository.insert_tool_audit(
        conversation_id=None, tool_call_id=None, tool_name="query_order",
        tool_source="builtin", mcp_server=None, arguments={"order_id": "1001"},
        result_summary="{}", status="成功", error_message=None,
        retry_count=0, duration_ms=12,
    )
    async with db_session_factory() as s:
        row = (await s.execute(select(ToolAuditLog))).scalars().one()
    assert row.tool_name == "query_order" and row.status == "成功"
    assert row.conversation_id is None          # 无会话上下文可空(表不挂外键)
    assert row.arguments == {"order_id": "1001"}


async def test_insert_tool_audit_all_statuses(db_session_factory):
    for st in ["成功", "失败", "超时", "校验拦下", "权限拒绝"]:
        await repository.insert_tool_audit(
            conversation_id=1, tool_call_id="tc-1", tool_name="create_ticket",
            tool_source="mcp", mcp_server="logistics", arguments=None,
            result_summary=None, status=st, error_message="原因",
            retry_count=2, duration_ms=None,
        )
    async with db_session_factory() as s:
        rows = (await s.execute(select(ToolAuditLog))).scalars().all()
    assert {r.status for r in rows} == {"成功", "失败", "超时", "校验拦下", "权限拒绝"}
