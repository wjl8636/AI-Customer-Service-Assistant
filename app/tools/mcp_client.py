# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch08 MCP Client:MultiServerMCPClient 多 Server 接入,现问现拿(每次现拉工具清单,
adapters 每次调用新建 session——Server 侧加工具,客服系统不重启即可见)。
权限/格式化只认我们侧:Server 自报的用途描述仅供模型参考,能不能调按 registry.WRITE_TOOLS。"""
import asyncio
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings
from app.tools import registry
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)


def _translate(mapping: dict[str, str], code):
    return mapping.get(code, code)


def _fmt_logistics(d: dict) -> dict:
    return {"tracking_no": d.get("tracking_no"),
            "status": _translate({"PICKED_UP": "已揽件", "IN_TRANSIT": "运输中",
                                  "DELIVERING": "派送中", "DELIVERED": "已签收"}, d.get("status_code")),
            "current_city": d.get("current_city"), "trace": d.get("trace")}


def _fmt_warranty(d: dict) -> dict:
    return {"order_id": d.get("order_id"),
            "warranty": _translate({"IN_WARRANTY": "在保", "EXPIRED": "已过保"}, d.get("warranty_code")),
            "warranty_until": d.get("warranty_until")}


def _fmt_return(d: dict) -> dict:
    return {"order_id": d.get("order_id"),
            "return_status": _translate({"AUDITING": "审核中", "RETURNING": "退货中",
                                         "REFUNDED": "已退款", "NONE": "无退货记录"}, d.get("return_code")),
            "updated_at": d.get("updated_at")}


# 结果格式化我们侧登记(挑回答用得上的字段 + 内部枚举码翻人话);未登记的 MCP 工具透传
FORMATTERS = {"query_logistics": _fmt_logistics, "query_warranty": _fmt_warranty,
              "query_return_status": _fmt_return}

_client: MultiServerMCPClient | None = None


def _connections() -> dict:
    return {
        "logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url},
        "aftersales": {"transport": "streamable_http", "url": settings.mcp_aftersales_url},
    }


def get_client() -> MultiServerMCPClient:
    global _client
    if _client is None:
        # handle_tool_errors=False:工具错误抛 ToolException,由执行引擎统一分诊/回灌
        _client = MultiServerMCPClient(_connections(), handle_tool_errors=False)
    return _client


async def _get_tools_of(*, server_name: str):
    """薄壳:单测 monkeypatch 锚点。"""
    return await get_client().get_tools(server_name=server_name)


async def fetch_mcp_specs() -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for server in _connections():
        try:
            # 我们侧超时封顶:连接拒绝会快速失败,但 Server 假死(TCP 接了不回话)只受 adapters
            # 默认超时保护——现问现拿每步都拉清单,最坏延迟必须封住
            tools = await asyncio.wait_for(_get_tools_of(server_name=server),
                                           timeout=settings.mcp_tool_timeout)
        except Exception as e:  # noqa: BLE001 单台不可达/假死:告警+跳过,不拖垮本轮对话
            logger.warning("MCP Server「%s」不可达,本轮跳过其工具:%s", server, type(e).__name__)
            continue
        for t in tools:
            specs.append(registry.spec_from_langchain_tool(
                t, source="mcp", mcp_server=server, format_result=FORMATTERS.get(t.name)))
    return specs
