# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import random
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from app.tools import registry
from app.tools.business import order_snapshot, owns_order

# 查不到和不是本人的单回同一句话。分开回就成了枚举 oracle:攻击者靠回答差异
# 就能挨个试出哪些订单号真实存在。
NOT_OWNED = {"error": "没有找到您的这笔订单", "code": "order_not_owned"}


@tool
async def query_order(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
    user_id: Annotated[str, InjectedToolArg],
) -> dict:
    """查询订单的状态、金额、下单时间、商品名和物流单号(tracking_no)。用于用户询问某个订单情况时。
    要查物流轨迹,需先用本工具拿到订单的 tracking_no,再把它传给 query_logistics。
    发起人身份由系统注入,你不要传 user_id。"""
    if not owns_order(user_id, order_id):
        return dict(NOT_OWNED)
    return order_snapshot(order_id)


@tool
async def query_product(
    product_name: Annotated[str, Field(description="商品名称或关键词,例如 猫粮")],
) -> dict:
    """查询商品的价格、库存和规格。用于用户咨询某商品是否有货、多少钱时。"""
    rng = random.Random(f"product:{product_name}")
    return {
        "product_name": product_name,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
        "spec": rng.choice(["标准装", "家庭装", "试用装"]),
    }


registry.register(registry.spec_from_langchain_tool(
    query_order, source="builtin", inject_user_id=True))
registry.register(registry.spec_from_langchain_tool(query_product, source="builtin"))
