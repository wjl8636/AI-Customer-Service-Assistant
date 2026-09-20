# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""售后 MCP Server(ch08 自建,mock 数据,不接真实系统、不建表)。
独立进程:uv run python mcp_servers/aftersales_server.py
工具:query_warranty(查在保)、query_return_status(查退货进度)。
MOCK_DELAY_SECONDS 同物流 Server,可注入延迟演示超时。"""
import asyncio
import os
import random
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("aftersales",
              host="127.0.0.1", port=int(os.environ.get("PORT", "8102")))

_DELAY = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))


@mcp.tool()
async def query_warranty(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """查询某订单商品是否在保修期内(在保状态、到期日)。用于用户问保修/在保时。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"warranty:{order_id}")
    code = rng.choice(["IN_WARRANTY", "EXPIRED"])         # 内部枚举码,client 侧翻人话
    return {"order_id": order_id, "warranty_code": code,
            "warranty_until": f"2026-{rng.randint(8, 12):02d}-{rng.randint(1, 28):02d}",
            "policy_ref": "AS-POLICY-07"}                 # 内部政策编号(回答用不上,client 侧应剔除)


@mcp.tool()
async def query_return_status(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """查询某订单的退货进度(审核中/退货中/已退款/无退货记录)。用于用户问退货到哪一步了。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"return:{order_id}")
    code = rng.choice(["AUDITING", "RETURNING", "REFUNDED", "NONE"])
    return {"order_id": order_id, "return_code": code,
            "updated_at": f"2026-07-{rng.randint(1, 16):02d} 10:00"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
