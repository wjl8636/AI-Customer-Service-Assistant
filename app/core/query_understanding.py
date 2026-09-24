import logging

from pydantic import BaseModel, Field

from app.core import llm
from app.core.llm import get_chat_model
from app.core.prompts import EXPAND_QUERIES_PROMPT, QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


class _Expanded(BaseModel):
    queries: list[str] = Field(default_factory=list, description="严格3条检索友好查询")


async def understand(query: str) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象),避开 glm-5.2 502。
    模型异常则回落原问题、扩展词留空:改写只是让检索更好命中,不是检索的前提条件,
    这一步挂了应该退化成「拿用户原话去查」,不该把整个请求带崩。"""
    model = llm.structured(_Rewrite)
    try:
        r: _Rewrite = await (QUERY_REWRITE_PROMPT | model).ainvoke({"query": query})
    except Exception:
        logger.warning("query 改写失败,回落原问题 query=%r", query, exc_info=True)
        return {"standard": query, "expanded": []}
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}


async def expand_queries(query: str) -> list[str]:
    """把一句问题泛化成检索友好查询(强制 JSON 单字段,扁平 list[str] 避开 glm 502)。
    严格取前 3 条非空;模型异常/全空则回落 [query],保证 retrieve_policy 至少有一条可查。
    只在 refund_flow 的 retrieve_policy 检索侧调(核心场景),商品咨询走 understand 的轻扩。"""
    model = llm.structured(_Expanded)
    try:
        r: _Expanded = await (EXPAND_QUERIES_PROMPT | model).ainvoke({"query": query})
        qs = [q.strip() for q in (r.queries or []) if q and q.strip()]
    except Exception:
        qs = []
    return qs[:3] or [query]
