# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch06 四验收端到端评估。需全服务起(mysql/milvus/上游/app)+ 已应用 sql/ch06-ticket-type.sql。
用法:.venv/bin/python -m scripts.eval_ch06"""
import asyncio

import httpx

BASE = "http://localhost:8000"


async def agent(client, msg, cid=None):
    r = await client.post(f"{BASE}/api/agent",
                          json={"user_id": "eval-ch06", "message": msg, "conversation_id": cid})
    return r.json()


async def main():
    async with httpx.AsyncClient(timeout=180) as c:
        results = []

        # 验收1:多轮 物流→退款→物流,每轮意图/指代随上下文(明细看 app 日志 intent/route/coref)
        r1 = await agent(c, "订单1001到哪了")
        cid = r1["conversation_id"]
        await agent(c, "那我想把它退了", cid)          # 指代「它」+ 意图漂移到退款退货
        await agent(c, "算了,它现在到哪了", cid)        # 又漂回物流
        results.append(("验收1 多轮意图漂移(看日志 intent/route)", True,
                        f"conv={cid};日志应见 route 物流→refund_flow→物流"))

        # 验收3:「这个能退吗」先补指代、再走退款子流程(带订单号,免 interrupt)
        b = await agent(c, "订单2002这个能退吗")
        acts = {a["type"] for a in b.get("suggested_actions", [])}
        results.append(("验收3 退款子流程(refund_form 或说明)", bool(b["answer"]),
                        {"actions": list(acts), "answer": b["answer"][:40]}))

        # 验收4(非流式部分):不带订单号问退款 → interrupt 返回订单列表
        b = await agent(c, "我要退款")
        results.append(("验收4 缺单弹订单选择器(interrupt)", bool(b.get("interrupt")),
                        b.get("interrupt")))

        for name, ok, detail in results:
            print(f"{'✅' if ok else '❌'} {name} -> {detail}")

    print("\n验收2(意图 JSON 稳定/怪问题落其他):跑 `.venv/bin/python -m scripts.eval_intent`。")
    print("验收4 前端 interrupt→resume→退款按钮 全链路:见 Task 15 浏览器截图。")
    print("验收1/3 的 coref/intent/route 明细:grep app 日志 `intent=.. route=.. trace={..coref..}`。")


if __name__ == "__main__":
    asyncio.run(main())
