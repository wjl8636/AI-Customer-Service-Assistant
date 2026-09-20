# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 自动化评估流水线:复用 ch04 评估集与指标,定期跑、落 eval_runs、连趋势。
指标:检索段 Recall@5 / MRR(hybrid_rerank,可答桶,跨文档题按组凑齐算),生成段 Faithfulness + D 桶拒答率。
运行:make eval-flywheel(TRIGGER=手动|定时,默认手动)。cron 示例:
  0 6 * * * cd /path/to/mewhelp && make eval-flywheel TRIGGER=定时 >> log/eval.log 2>&1
趋势:读最近 10 轮,对比上一轮涨跌;任何指标下滑标 ⚠——README:「重点全在这条趋势线上」。
产物:data/ch09/reports/eval_trend.txt(趋势本身的权威源是 eval_runs 表,
「观测与成本」页直接读表,不读这份文本,免得同一条趋势有两个出处)。
另落 eval_trend_note.json:模型看着这一轮的涨跌写的读图小注,带上它描述的轮次 id,
页面只在这个 id 还是最新一轮时才用它。
"""
import argparse
import asyncio
import json
import pathlib
import sys

from pydantic import BaseModel, Field

from app.core import llm, read_notes
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.db import repository
from scripts.eval_ch04 import (
    GRADED_BUCKETS,
    RECALL_K,
    _format_evidence,
    _load,
    _mean,
    _ranks_per_group,
    _recall_at,
    _refusal_one,
    _retrieve,
    _rr,
    _try,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "data/ch09/reports/eval_trend.txt"
_NOTE = _ROOT / "data/ch09/reports/eval_trend_note.json"
_METRICS = ("recall_at_5", "mrr", "faithfulness", "refusal_rate")
_LINES: list[str] = []


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


class _Faith(BaseModel):
    faithful: bool = Field(description="是否忠实于证据")
    reason: str = Field(default="")


async def _gen_faith(s, hits, model, judge):
    evidence = _format_evidence(hits)
    ans = await _try(lambda: (RAG_ANSWER_PROMPT | model).ainvoke(
        {"query": s["query"], "evidence": evidence}), f"gen:{s['id']}")
    if ans is None:
        return None
    text = ans.content if isinstance(ans.content, str) else str(ans.content)
    fa = await _try(lambda: (FAITHFULNESS_PROMPT | judge).ainvoke(
        {"evidence": evidence, "answer": text}), f"faith:{s['id']}")
    return None if fa is None else bool(fa.faithful)


async def run_once() -> dict:
    samples = _load()
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    absent = [s for s in samples if s["bucket"] == "D_absent"]

    hits_list = await asyncio.gather(*(_retrieve("hybrid_rerank", s) for s in graded))
    # 口径跟 ch04 完全一致:跨文档题按组算,凑齐才算召回到(直接复用那边的函数)
    ranks = [_ranks_per_group(h, s) for s, h in zip(graded, hits_list)]
    recall = _mean([_recall_at(r, RECALL_K) for r in ranks])
    mrr = _mean([_rr(r) for r in ranks])

    model = get_chat_model()
    judge = llm.structured(_Faith)
    faiths = await asyncio.gather(*(
        _gen_faith(s, h, model, judge) for s, h in zip(graded, hits_list)))
    faithfulness = _mean([1.0 if f else 0.0 for f in faiths if f is not None])

    refusals = await asyncio.gather(*(_refusal_one(s) for s in absent))
    got = [r["refused"] for r in refusals if isinstance(r, dict) and "refused" in r]
    refusal_rate = sum(got) / len(got) if got else 0.0

    return {"dataset_size": len(samples),
            "metrics": {"recall_at_5": round(recall, 3), "mrr": round(mrr, 3),
                        "faithfulness": round(faithfulness, 3),
                        "refusal_rate": round(refusal_rate, 3)}}


async def _trend_note(runs) -> None:
    """趋势的读图小注另存一份,并写上它描述的那一轮 id。

    表才是趋势的权威源,注只能是旁挂的一份;不把 id 记下来,下一轮跑完但注没生成
    (上游不稳)时,页面就会拿上一轮的话去解释这一轮的数。
    """
    latest = runs[0]
    prev = runs[1] if len(runs) > 1 else None
    lm, pm = latest.metrics or {}, (prev.metrics or {}) if prev else {}
    zh = {"recall_at_5": "Recall@5", "recall_at_10": "Recall@10", "mrr": "MRR",
          "faithfulness": "忠实度", "refusal_rate": "库外拒答率"}
    payload = {
        "本轮": {"轮次": latest.id, "触发": latest.triggered_by,
                 **{zh[n]: lm.get(n) for n in _METRICS}},
        "上一轮": {"轮次": prev.id, **{zh[n]: pm.get(n) for n in _METRICS}} if prev else None,
        "涨跌": {zh[n]: round(lm.get(n, 0) - pm.get(n, 0), 3) for n in _METRICS} if prev else None,
    }
    note = await read_notes.generate("eval_trend", payload)
    _log("\n读图小注:" + (note if note else "本轮没有(页面用兜底句)"))
    _NOTE.write_text(json.dumps({"run_id": latest.id, "note": note},
                                ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _trend(runs) -> None:
    """runs 按新→旧;打印每轮各指标,并与上一轮(更旧一行)对比涨跌,下滑标 ⚠。"""
    _log("\n=== 评估趋势(新在上)===")
    names = list(_METRICS)
    _log(f"{'轮次':>4s} {'时间':14s} {'触发':4s}" + "".join(f"{n:>18s}" for n in names))
    for i, r in enumerate(runs):
        prev = runs[i + 1].metrics if i + 1 < len(runs) else None
        row = f"#{r.id:>3d} {r.created_at:%m-%d %H:%M}  {r.triggered_by:4s}"
        for n in names:
            v = r.metrics.get(n)
            cell = f"{v:.3f}" if v is not None else "—"
            if prev and v is not None and prev.get(n) is not None:
                d = v - prev[n]
                cell += " ↑" if d > 0.005 else (" ⚠↓" if d < -0.005 else " →")
            row += f"{cell:>18s}"
        _log(row)
    if len(runs) > 1:
        latest, older = runs[0].metrics, runs[1].metrics
        drops = [n for n in names if latest.get(n, 0) < older.get(n, 0) - 0.005]
        _log(f"\n⚠ 下滑指标:{', '.join(drops)}" if drops else "\n所有指标持平或上涨。")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggered-by", default="手动", choices=["手动", "定时"])
    args = ap.parse_args()

    _log(f"=== ch09 评估流水线(触发:{args.triggered_by})===")
    result = await run_once()
    m = result["metrics"]
    _log(f"本轮:Recall@5={m['recall_at_5']:.3f} MRR={m['mrr']:.3f} "
         f"Faithfulness={m['faithfulness']:.3f} 拒答率={m['refusal_rate']:.3f}")
    await repository.insert_eval_run(args.triggered_by, result["dataset_size"], m)
    runs = await repository.list_eval_runs(limit=10)
    _trend(runs)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    await _trend_note(runs)
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _log(f"\n趋势报告已落 {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
