from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from app.tools import registry
from app.tools.builtin.orders import NOT_OWNED
from app.tools.business import owns_order


@tool
async def submit_refund(
    order_id: Annotated[str, Field(description="要退款的订单号")],
    user_id: Annotated[str, InjectedToolArg],
    reason: Annotated[str | None, Field(description="退款原因(可选,最终以前端固定类目下拉为准)")] = None,
) -> dict:
    """判定这一单可以退款后,调用本工具发起退款申请。实际提交由前端退款表单确认后落库,
    本工具只表示『这一单可以退,已把提交入口交给用户』。
    发起人身份由系统注入,你不要传 user_id。"""
    # 写操作有二次确认门,但那道门确认的是「要不要退」,不是「这单是不是你的」,归属得单独校验
    if not owns_order(user_id, order_id):
        return dict(NOT_OWNED)
    return {"status": "待用户确认", "order_id": order_id}


registry.register(registry.spec_from_langchain_tool(
    submit_refund, source="builtin", inject_user_id=True))
