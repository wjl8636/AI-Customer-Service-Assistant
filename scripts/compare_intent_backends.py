# 意图识别后端测评:普通LLM(结构化,默认) vs Jev(typesafe.ai 决策模型 Choice)。
# 指标:token 消耗(input/output)、单次处理耗时、九类意图识别准确率(含每类明细 + 混淆)。
# 测评数据分布均衡:九类意图各 56 条(共 504 条,无上下文),另 + 多条带上下文的漂移用例单独统计。
# 数据集在 scripts/intent_dataset.py。
#
# 用法(.env 里配好 JEV_API_KEY/INTENT_BACKEND 或环境变量导出):
#   PYTHONPATH=. uv run python scripts/compare_intent_backends.py            # 两后端都测并对比
#   PYTHONPATH=. uv run python scripts/compare_intent_backends.py --llm      # 只测普通LLM
#   PYTHONPATH=. uv run python scripts/compare_intent_backends.py --jev      # 只测Jev
# 报告落 data/intent_backend/report.{json,txt}。
import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.core import intent
from intent_dataset import DATASET, DRIFT

LABELS = intent.INTENTS  # 九类,顺序一致


def _check_balanced() -> None:
    from collections import Counter
    c = Counter(lab for _, lab in DATASET)
    assert set(c) == set(LABELS), f"缺失意图类:{set(LABELS) - set(c)}"
    assert len(set(c.values())) == 1, f"各类数量不均:{dict(c)}"
    print(f"均衡集:九类各 {c[LABELS[0]]} 条,共 {len(DATASET)} 条;漂移用例 {len(DRIFT)} 条")


async def _run_one(rfn, query: str, label: str, history: str) -> dict:
    t0 = time.perf_counter()
    r = await rfn(query, history)
    return {"query": query, "gold": label, "got": r["intent"], "ok": r["intent"] == label,
            "confidence": r["confidence"],
            "latency_ms": r["latency_ms"] if "latency_ms" in r else (time.perf_counter() - t0) * 1000,
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}


async def run_backend(name: str) -> tuple[list[dict], list[dict]]:
    settings.intent_backend = name
    rfn = intent.classify_with_usage
    core = [await _run_one(rfn, q, lab, "") for q, lab in DATASET]
    drift = [await _run_one(rfn, q, lab, hist) for q, lab, hist in DRIFT]
    return core, drift


def _agg(core_rows: list[dict]) -> dict:
    n = len(core_rows)
    ok = sum(r["ok"] for r in core_rows)
    lat = [r["latency_ms"] for r in core_rows]
    intr = [r["input_tokens"] for r in core_rows]
    outr = [r["output_tokens"] for r in core_rows]
    per_class = {}
    for lab in LABELS:
        sub = [r for r in core_rows if r["gold"] == lab]
        if sub:
            per_class[lab] = {"hits": sum(r["ok"] for r in sub), "total": len(sub),
                              "acc": sum(r["ok"] for r in sub) / len(sub)}
    return {
        "overall_acc": ok / n, "hits": ok, "total": n,
        "latency_ms_avg": round(statistics.mean(lat), 1),
        "latency_ms_p50": round(statistics.median(lat) if lat else 0, 1),
        "latency_ms_p95": round(sorted(lat)[int(n * 0.95) - 1] if lat else 0, 1),
        "input_tokens_sum": sum(intr), "output_tokens_sum": sum(outr),
        "total_tokens": sum(intr) + sum(outr),
        "tokens_per_call_avg": round((sum(intr) + sum(outr)) / n, 1),
        "per_class": per_class,
    }


def _confusion(core_rows: list[dict]) -> dict[str, dict[str, int]]:
    m = {lab: {l2: 0 for l2 in LABELS} for lab in LABELS}
    for r in core_rows:
        m[r["gold"]][r["got"]] += 1
    return m


def _fmt_cells(head, rows, width=14):
    # 简单打印对齐表
    return head, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="意图识别后端测评:普通LLM vs Jev")
    ap.add_argument("--llm", action="store_true", help="只测普通LLM后端")
    ap.add_argument("--jev", action="store_true", help="只测Jev后端")
    args = ap.parse_args()
    want = []
    if args.llm and not args.jev:
        want = ["llm"]
    elif args.jev and not args.llm:
        want = ["jev"]
    else:
        want = ["llm", "jev"]   # 默认都测

    _check_balanced()
    if "jev" in want and not settings.jev_api_key:
        print("[!] 未检测到 JEV_API_KEY,Jev 测评会全部落入兜底「其他」;请在 .env 或环境变量设置后再测 Jev。")

    results: dict = {"data_balanced": True, "classes": list(LABELS),
                     "samples_per_class": len(DATASET) // len(LABELS),
                     "run_at": datetime.now().isoformat(timespec="seconds"),
                     "backends": {}}
    for name in want:
        print(f"\n===== 后端: {name} =====")
        core, drift = asyncio.run(run_backend(name))
        agg = _agg(core)
        confusion = _confusion(core)
        # 漂移准确率
        drift_acc = sum(r["ok"] for r in drift) / len(drift) if drift else 0.0

        results["backends"][name] = {"agg": agg, "confusion": confusion,
                                     "drift_acc": round(drift_acc, 4),
                                     "drift_samples": drift,
                                     "core_samples": core}

        print(f"均衡集 {agg['hits']}/{agg['total']} → 准确率 {agg['overall_acc']:.2%}  (漂移 {drift_acc:.2%})")
        print(f"耗时: avg {agg['latency_ms_avg']}ms / p50 {agg['latency_ms_p50']}ms / p95 {agg['latency_ms_p95']}ms")
        print(f"token: 输入 {agg['input_tokens_sum']} + 输出 {agg['output_tokens_sum']} "
              f"= 合计 {agg['total_tokens']} (平均 {agg['tokens_per_call_avg']}/次)")
        print("每类准确率: " + "  ".join(
            f"{lab}={p['acc']:.0%}" for lab, p in agg["per_class"].items()))
        _print_failures(core)
        print("混淆矩阵(gold 行 → 预测列;对角=对):")
        print("        " + " ".join(f"{l:>6}" for l in LABELS))
        for gold in LABELS:
            print(f"{gold:>6}: " + " ".join(f"{confusion[gold][p]:>6}" for p in LABELS))

    _write_report(results)

    # 对比摘要
    if {"llm", "jev"} <= set(results["backends"]):
        a, b = results["backends"]["llm"]["agg"], results["backends"]["jev"]["agg"]
        print("\n========== 对比摘要 ==========")
        print(f"{'指标':<24}{'LLM':>14}{'Jev':>14}{'差异':>16}")
        print(f"{'准确率':<24}{a['overall_acc']:>12.2%}{b['overall_acc']:>12.2%}{b['overall_acc']-a['overall_acc']:>+14.2%}")
        print(f"{'平均耗时(ms)':<22}{a['latency_ms_avg']:>14.1f}{b['latency_ms_avg']:>14.1f}{b['latency_ms_avg']/a['latency_ms_avg'] if a['latency_ms_avg'] else 0:>14.2f}x")
        print(f"{'input token/次':<22}{a['input_tokens_sum']/a['total']:>14.1f}{b['input_tokens_sum']/b['total']:>14.1f}")
        print(f"{'output token/次':<22}{a['output_tokens_sum']/a['total']:>14.1f}{b['output_tokens_sum']/b['total']:>14.1f}")
        print(f"{'总 token/次':<22}{a['tokens_per_call_avg']:>14.1f}{b['tokens_per_call_avg']:>14.1f}{b['tokens_per_call_avg']/a['tokens_per_call_avg'] if a['tokens_per_call_avg'] else 0:>14.2f}x")


def _print_failures(core_rows: list[dict]) -> None:
    bad = [r for r in core_rows if not r["ok"]]
    if bad:
        print(f"{len(bad)} 条判错:")
        for r in bad:
            print(f"  ❌ {r['query']!r} 期望={r['gold']} 得={r['got']}")
    else:
        print("全部判对 ✅")


def _write_report(results: dict) -> None:
    out_dir = Path("data") / "intent_backend"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()