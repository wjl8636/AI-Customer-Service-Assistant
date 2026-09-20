# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""标注样例评估:核对 glm-5.2 是否按预期选中工具,并观察 query_faq 的关键词与漏召回。
需服务运行中(docker compose/mewhelp-mysql + make seed + make dev)。"""
import asyncio
import json

import httpx

BASE = "http://localhost:8000"

# (用户问法, 期望命中的工具名集合;None 表示期望不调用任何工具)
SAMPLES = [
    ("订单 1001 的物流到哪了", {"query_logistics"}),
    ("退货政策是什么", {"query_faq"}),          # 验收2:query_faq 命中
    ("买的鞋子能不能退", {"query_faq"}),          # 模型提「退货」→ 命中「退货政策」,未漏(语义提参救了)
    ("邮费是多少", {"query_faq"}),               # 验收3:模型提「邮费」→ LIKE question 漏,但答案在「运费怎么算」→ 语义鸿沟,留 ch03
    ("iPhone 还有货吗", {"query_product"}),
    ("订单 2002 多少钱", {"query_order"}),
    ("我要投诉,给我登记一下", {"create_ticket"}),
    ("今天天气怎么样", None),                       # 超范围,期望不调工具
]


def _faq_note(body: dict) -> str:
    """对调用了 query_faq 的样例,打印它提取的 keyword 与是否漏召回。"""
    faq_calls = [tc for tc in body["tool_calls"] if tc["name"] == "query_faq"]
    faq_results = [tr for tr in body["tool_results"] if tr["name"] == "query_faq"]
    if not faq_calls or not faq_results:
        return ""
    keyword = faq_calls[0]["args"].get("keyword", "")
    hits = json.loads(faq_results[0]["content"]).get("hits", [])
    return f"  [query_faq keyword={keyword!r} 漏召回={'是' if not hits else '否'}]"


async def main() -> None:
    passed = 0
    async with httpx.AsyncClient(timeout=60) as client:
        for msg, expect in SAMPLES:
            r = await client.post(f"{BASE}/api/agent", json={"user_id": "eval", "message": msg})
            body = r.json()
            names = {tc["name"] for tc in body["tool_calls"]}
            ok = (not names) if expect is None else bool(names & expect)
            passed += ok
            print(f"{'✅' if ok else '❌'} {msg!r} -> {names or '(未调用)'} 期望={expect}{_faq_note(body)}")
            print(f"    answer: {body['answer'][:70]}")
    print(f"\n选对工具 {passed}/{len(SAMPLES)}(glm 非确定性,抖动如实重跑记录)")


if __name__ == "__main__":
    asyncio.run(main())
