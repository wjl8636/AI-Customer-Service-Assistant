# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch08 验收样例端到端评估:建工单确认流(追问/预览/确认/取消)+ 物流走 MCP。
需全服务起:make dev(含两台 MCP Server)+ MySQL + 聊天上游;已应用 sql/ch08-ddl.sql。
用法:.venv/bin/python -m scripts.eval_ch08
判据即验收标准:中断有无/工单落表/审计状态/工具轨迹;追问措辞打印出来供人工复核(prompt 类产出)。"""
import asyncio
import json

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings

BASE = "http://localhost:8000"


async def agent(client, msg, cid=None):
    r = await client.post(f"{BASE}/api/agent",
                          json={"user_id": "eval-ch08", "message": msg, "conversation_id": cid})
    r.raise_for_status()
    return r.json()


async def resume(client, cid, confirmed: bool) -> str:
    """POST /api/actions/resume(SSE),拼接 delta 为最终答复文本。"""
    answer = ""
    async with client.stream("POST", f"{BASE}/api/actions/resume",
                             json={"conversation_id": cid, "confirmed": confirmed}) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data.get("delta") is not None:
                answer += data["delta"]
    return answer


async def db_scalar(engine, sql: str):
    async with engine.connect() as conn:
        return (await conn.execute(text(sql))).scalar()


async def main():
    engine = create_async_engine(settings.database_url)
    results = []

    async with httpx.AsyncClient(timeout=180) as c:
        # ---- 验收4a:明确要建单但没讲清楚问题 → 追问,不弹卡不落单 ----
        tickets_before = await db_scalar(engine, "SELECT COUNT(*) FROM tickets")
        r1 = await agent(c, "帮我建个工单")
        cid = r1["conversation_id"]
        tickets_now = await db_scalar(engine, "SELECT COUNT(*) FROM tickets")
        no_interrupt = r1.get("interrupt") is None
        results.append(("验收4a 缺问题描述先追问(无预览卡/无落单)",
                        no_interrupt and tickets_now == tickets_before,
                        f"answer={r1['answer']!r}"))

        # ---- 验收4b:补齐问题描述 → 弹工单预览卡(interrupt confirm_ticket) ----
        r2 = await agent(c, "我的智能猫砂盆漏电了,一开机就跳闸", cid)
        intr = r2.get("interrupt") or {}
        ok_preview = intr.get("type") == "confirm_ticket" and bool(
            (intr.get("preview") or {}).get("description"))
        results.append(("验收4b 补齐后弹工单预览卡(type/preview 齐)", ok_preview,
                        f"interrupt={intr}"))

        # ---- 验收4c:确认提交 → tickets 落一条,答复带工单号 ----
        answer = await resume(c, cid, confirmed=True)
        ticket_no = await db_scalar(
            engine, "SELECT ticket_no FROM tickets ORDER BY created_at DESC, ticket_no DESC LIMIT 1")
        tickets_after = await db_scalar(engine, "SELECT COUNT(*) FROM tickets")
        audit_ok = await db_scalar(
            engine, "SELECT status FROM tool_audit_logs WHERE tool_name='create_ticket' "
                    "ORDER BY id DESC LIMIT 1")
        results.append(("验收4c 确认提交→落 tickets+审计成功+答复带工单号",
                        tickets_after == tickets_before + 1 and audit_ok == "成功"
                        and bool(ticket_no) and ticket_no in answer,
                        f"ticket_no={ticket_no} 审计={audit_ok} answer={answer!r}"))

        # ---- 验收5:走到预览卡后点取消 → 不建单,审计「权限拒绝」 ----
        r3 = await agent(c, "帮我建个工单,我的猫爬架第三层塌了")
        cid2 = r3["conversation_id"]
        intr3 = r3.get("interrupt") or {}
        answer3 = ""
        if intr3.get("type") == "confirm_ticket":
            answer3 = await resume(c, cid2, confirmed=False)
        tickets_final = await db_scalar(engine, "SELECT COUNT(*) FROM tickets")
        audit_deny = await db_scalar(
            engine, "SELECT status FROM tool_audit_logs WHERE tool_name='create_ticket' "
                    "ORDER BY id DESC LIMIT 1")
        results.append(("验收5 取消→不建单+审计「权限拒绝」",
                        intr3.get("type") == "confirm_ticket"
                        and tickets_final == tickets_after and audit_deny == "权限拒绝",
                        f"interrupt={intr3.get('type')} 审计={audit_deny} answer={answer3!r}"))

        # ---- 验收2:物流轨迹由物流 MCP Server 接管(工具轨迹+审计来源) ----
        r4 = await agent(c, "订单1001的物流到哪了")
        names = [tc["name"] for tc in r4.get("tool_calls", [])]
        lg = [tr for tr in r4.get("tool_results", []) if tr["name"] == "query_logistics"]
        zh_status = any(s in (lg[0]["content"] if lg else "")
                        for s in ("已揽件", "运输中", "派送中", "已签收"))
        src = await db_scalar(
            engine, "SELECT CONCAT(tool_source,'/',IFNULL(mcp_server,'')) FROM tool_audit_logs "
                    "WHERE tool_name='query_logistics' ORDER BY id DESC LIMIT 1")
        results.append(("验收2 物流走 MCP(轨迹含中文状态+审计来源 mcp/logistics)",
                        "query_logistics" in names and zh_status and src == "mcp/logistics",
                        f"tools={names} 审计来源={src} answer={r4['answer']!r}"))

    await engine.dispose()
    print("\n" + "=" * 72)
    failed = 0
    for name, ok, detail in results:
        print(f"{'✅' if ok else '❌'} {name}\n   {detail}")
        failed += not ok
    print(f"\n{len(results) - failed}/{len(results)} PASS")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
