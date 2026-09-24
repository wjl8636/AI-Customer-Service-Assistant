"""ch05 五验收端到端评估。需全服务起(mysql/milvus/上游/app 新代码)。
用法:.venv/bin/python -m scripts.eval_ch05
走 /api/agent(非流式,看得到工具轨迹与 suggested_actions);验收1 看 app 日志、验收3 前端部分见浏览器。"""
import asyncio

import httpx

BASE = "http://localhost:8000"


async def agent(client, msg):
    """每条用独立会话(conversation_id=None),避免跨题上下文污染。"""
    r = await client.post(
        f"{BASE}/api/agent",
        json={"user_id": "eval-ch05", "message": msg, "conversation_id": None},
    )
    return r.json()


async def main() -> None:
    async with httpx.AsyncClient(timeout=120) as c:
        results = []

        # 验收2:业务数据类,Agent 自己调工具(物流依赖订单 → query_order + query_logistics)
        b = await agent(c, "订单1001的物流到哪了")
        names = {tc["name"] for tc in b["tool_calls"]}
        results.append(("验收2 物流自调工具", "query_logistics" in names, sorted(names)))

        # 验收3(后端部分):投诉出「转人工」「建工单」两可选项,后端不自动建单
        b = await agent(c, "我要投诉,你们太差了")
        types = {a["type"] for a in b.get("suggested_actions", [])}
        results.append(("验收3 投诉出两可选项", {"transfer_human", "create_ticket"} <= types, sorted(types)))

        # 验收4:闲聊固定话术,零工具
        b = await agent(c, "你好呀")
        results.append(("验收4 闲聊固定话术零工具", not b["tool_calls"] and bool(b["answer"]), b["answer"][:24]))

        # 验收5:复杂问 → 真·顺序多步(query_logistics 需 query_order 产出的 tracking_no)
        b = await agent(c, "我手机尾号1001那个订单发货没?到哪了?")
        names = {tc["name"] for tc in b["tool_calls"]}
        chained = {"query_order", "query_logistics"} <= names
        results.append(("验收5 ReAct 顺序多步(order→logistics 链)", chained, sorted(names)))

    print("=" * 60)
    for name, ok, detail in results:
        print(f"{'✅' if ok else '❌'} {name} -> {detail}")
    print("=" * 60)
    print("验收1(强制检索节点被走到):问「退货政策是什么」后看 app 日志,")
    print("  应见 `ch05 turn ... route=knowledge trace={... 'forced_rag': True ...}`")
    print("验收3(前端):浏览器点「转人工」显示已转接+客服小猫问候;点「建工单」写 tickets;都不点继续正常对话")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n后端可断言项 {passed}/{len(results)} 通过(glm/gpt 非确定性,抖动如实重跑记录)")


if __name__ == "__main__":
    asyncio.run(main())
