# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""Query 扩写标注评估:核心场景出 3 条 + JSON 稳定 + 保持关键实体。需聊天上游可调通。
用法:.venv/bin/python -m scripts.eval_expand"""
import asyncio

from app.core.query_understanding import expand_queries

CASES = ["蓝牙耳机还能申请退货吗", "订单1001的猫粮质量问题能退款吗", "智能猫砂盆坏了怎么保修"]


async def main():
    for q in CASES:
        qs = await expand_queries(q)
        print(f"{'✅' if len(qs) == 3 else '⚠️'} {q!r} -> {qs}(条数={len(qs)})")
    print("\n核心场景应出 3 条、彼此侧重点不同、含原问题关键实体;JSON 不稳时条数会 <3。抖动如实重跑记录")


if __name__ == "__main__":
    asyncio.run(main())
