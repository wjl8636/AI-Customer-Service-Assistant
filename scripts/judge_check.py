# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""裁判回归:拿人工处置过的编造个案台账,反过来考忠实度裁判。

台账里每条个案都被人看过一遍并留了处置:
- 「已解决」= 人工确认真编了(库里当时没有这句),裁判判 false 才对;
- 「无需解决」= 人工看过是裁判判严了,裁判应该判 true;
- 「未解决」= 还没过目,没有标准答案,跳过。

个案存了当时喂给模型的 Top-K 证据全集(citations)和当时那版答案,所以这里能**原样重放**
——不重新检索、不重新生成,只换裁判。裁判提示词改一句,跑一次就知道是修对了还是把真编造也放过了。

用法:make judge-check(需要台账里有已处置的个案;评估集自检走 make eval-check)。
一致率不到 100% 时退出码 1:裁判和人工的判断对不上,要么提示词还得改,要么这条处置该复议。
"""
import asyncio
import sys

from app.core import llm
from app.core.prompts import FAITHFULNESS_PROMPT
from app.db import repository
from pydantic import BaseModel, Field

CONCURRENCY = 5


class _Faith(BaseModel):
    faithful: bool = Field(description="是否忠实于证据")
    reason: str = Field(description="一句话理由")


def evidence_from_citations(citations) -> str:
    """把台账里的角标原文拼回评估时那份证据文本(格式与 eval_ch04._format_evidence 一致)。"""
    return "\n".join(f"[{c.get('n', i + 1)}] {c.get('question', '')}:{c.get('answer', '')}"
                     for i, c in enumerate(citations or []))


def expected_faithful(status: str) -> bool | None:
    """人工处置 → 裁判该给的答案。未解决的没标准答案,返回 None 表示跳过。"""
    return {"已解决": False, "无需解决": True}.get(status)


def gradable(rows) -> list:
    """能考的个案:有人工处置、且当时留了角标原文(老个案没快照,重放不出来)。"""
    return [r for r in rows
            if expected_faithful(r.status) is not None and (r.citations or [])]


def verdict_row(case_id: str, status: str, faithful: bool | None) -> dict:
    exp = expected_faithful(status)
    return {"id": case_id, "status": status, "expected": exp, "actual": faithful,
            "agree": faithful is not None and faithful == exp}


async def _judge_one(row, judge, sem) -> dict:
    # 重试一次:这条通道偶发回一个空 completion(解析成 None),记成「调用失败」会被误读成
    # 「裁判判反了」。同 eval_ch04._try 的处理——先退一步重试,仍失败才记账
    async with sem:
        for attempt in (1, 2):
            try:
                fa = await (FAITHFULNESS_PROMPT | judge).ainvoke(
                    {"evidence": evidence_from_citations(row.citations), "answer": row.answer})
                out = verdict_row(row.eval_id, row.status, bool(fa.faithful))
                out["reason"] = fa.reason
                return out
            except Exception as e:
                err = f"裁判调用失败:{type(e).__name__}"
                if attempt == 2:
                    out = verdict_row(row.eval_id, row.status, None)
                    out["reason"] = err
                    return out
                await asyncio.sleep(1)


async def main() -> int:
    rows, total, counts = await repository.list_faith_cases(None, 1, 200)
    cases = gradable(rows)
    if not cases:
        print(f"台账 {total} 条,没有「已处置 + 有角标原文」的个案,考不了。先在 RAG 评估页处置几条。")
        return 0

    judge = llm.structured(_Faith, temperature=0)   # 尺子不能有抖动,和评估里的裁判同一档
    sem = asyncio.Semaphore(CONCURRENCY)
    outs = await asyncio.gather(*(_judge_one(r, judge, sem) for r in cases))
    outs.sort(key=lambda o: o["id"])

    label = {True: "忠实", False: "编造", None: "调用失败"}
    print(f"裁判回归:台账 {total} 条,能考 {len(cases)} 条(处置分布 {counts})\n")
    print(f"{'个案':6s} {'人工处置':8s} {'人工期望':6s} {'裁判这次':8s} 结果")
    for o in outs:
        print(f"{o['id']:6s} {o['status']:8s} {label[o['expected']]:6s} "
              f"{label[o['actual']]:8s} {'一致' if o['agree'] else '✗ 不一致'}")
    agree = sum(1 for o in outs if o["agree"])
    print(f"\n一致 {agree}/{len(outs)}({agree / len(outs):.0%})")
    for o in outs:
        if not o["agree"]:
            print(f"  {o['id']} 裁判理由:{o['reason']}")
    return 0 if agree == len(outs) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
