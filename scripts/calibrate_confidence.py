# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 置信度阈值校准:拿 ch04 评估集,A/B/C 桶(应可答)与 D 桶(应拒答)分别算
evidence_confidence 分布,扫阈值取 Youden J(=可答通过率 - 应拒放行率)最大的分离点。
输出分布表 + 推荐阈值 → 人工回填 settings.evidence_confidence_threshold(不拍脑袋)。
运行:make calibrate-confidence(需 milvus + 上游可调通 + 知识库已建)。
产物:data/ch09/reports/confidence_calibration.{txt,json};「观测与成本」页只读这份 json。
"""
import asyncio
import json
import pathlib
import sys
from datetime import datetime

from app.core import read_notes, retrieval
from app.core.confidence import compute_evidence_confidence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data/ch09/reports"
_OUT = _OUT_DIR / "confidence_calibration.txt"
_OUT_JSON = _OUT_DIR / "confidence_calibration.json"
ANSWERABLE = {"A_policy", "B_model", "C_colloquial", "E_multi"}
_SEM = asyncio.Semaphore(8)
_LINES: list[str] = []


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


async def _conf(sample) -> tuple[str, float]:
    async with _SEM:
        hits = await retrieval.search_knowledge(sample["query"], strategy="hybrid_rerank")
    return sample["bucket"], compute_evidence_confidence(hits).score


def _dist(name, xs) -> dict | None:
    """打印一行分布,同时把这行的数原样返回给 json 产物——两边同源,不各算一遍。"""
    xs = sorted(xs)
    if not xs:
        return None
    def p(q):
        return xs[min(len(xs) - 1, int(q * len(xs)))]
    stat = {"n": len(xs), "min": round(xs[0], 3), "p25": round(p(.25), 3),
            "p50": round(p(.5), 3), "p75": round(p(.75), 3), "max": round(xs[-1], 3)}
    _log(f"{name:12s} n={stat['n']:3d} min={stat['min']:.3f} p25={stat['p25']:.3f} "
         f"p50={stat['p50']:.3f} p75={stat['p75']:.3f} max={stat['max']:.3f}")
    return stat


async def main():
    samples = [json.loads(ln) for ln in open("tests/data/eval_ch04.jsonl")]
    results = await asyncio.gather(*(_conf(s) for s in samples))
    answerable = [c for b, c in results if b in ANSWERABLE]
    absent = [c for b, c in results if b == "D_absent"]

    _log("=== evidence_confidence 分布(hybrid_rerank)===")
    dist = {"answerable": _dist("可答(ABCE)", answerable), "absent": _dist("应拒(D)", absent)}

    _log("\n=== 阈值扫描(通过率=conf>=t 的可答占比;放行率=conf>=t 的应拒占比)===")
    _log(f"{'t':>6s} {'可答通过率':>10s} {'应拒放行率':>10s} {'YoudenJ':>8s}")
    scan, best_t, best_j = [], 0.0, -1.0
    for i in range(5, 96):
        t = i / 100
        tpr = sum(1 for c in answerable if c >= t) / len(answerable)
        fpr = sum(1 for c in absent if c >= t) / len(absent)
        j = tpr - fpr
        scan.append({"t": round(t, 2), "pass_rate": round(tpr, 3),
                     "leak_rate": round(fpr, 3), "youden_j": round(j, 3)})
        if i % 5 == 0 or j > best_j:
            _log(f"{t:6.2f} {tpr:10.3f} {fpr:10.3f} {j:8.3f}")
        if j > best_j:
            best_t, best_j = t, j
    _log(f"\n推荐阈值 evidence_confidence_threshold = {best_t:.2f}(Youden J={best_j:.3f})")
    _log("请回填 app/config.py 默认值,并在 dev-notes/ch09.md 记录本次校准。")

    pick = next((r for r in scan if r["t"] == round(best_t, 2)), None)
    recommended = {"threshold": round(best_t, 2), "youden_j": round(best_j, 3),
                   "pass_rate": pick["pass_rate"] if pick else None,
                   "leak_rate": pick["leak_rate"] if pick else None}
    # 读图小注只喂十分位那几条扫描线:整条 91 点曲线塞进去,模型容易挑个无关的点来说事。
    # 字段名一律换成页面上那几个中文表头,注里就不会出现 pass_rate 这种说法
    def _zh(stat):
        return None if not stat else {"题数": stat["n"], "最低": stat["min"], "p25": stat["p25"],
                                      "中位": stat["p50"], "p75": stat["p75"], "最高": stat["max"]}
    note = await read_notes.generate("confidence_calibration", {
        "证据置信度分布": {"可答(政策/型号/口语/跨文档)": _zh(dist["answerable"]),
                          "应拒(库外)": _zh(dist["absent"])},
        "选定的线": {"阈值": recommended["threshold"], "可答通过率": recommended["pass_rate"],
                    "应拒放行率": recommended["leak_rate"], "YoudenJ": recommended["youden_j"]},
        "阈值扫描(每 0.1 一条)": [
            {"阈值": r["t"], "可答通过率": r["pass_rate"], "应拒放行率": r["leak_rate"]}
            for r in scan if round(r["t"] * 10, 3) == int(r["t"] * 10)]})
    _log("\n读图小注:" + (note if note else "本轮没有(页面用兜底句)"))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _OUT_JSON.write_text(json.dumps({
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "strategy": "hybrid_rerank", "dataset": "tests/data/eval_ch04.jsonl",
                 "n_answerable": len(answerable), "n_absent": len(absent)},
        "distribution": dist,
        "scan": scan,
        "recommended": recommended,
        "read_notes": {"confidence_calibration": note} if note else {},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"报告已落 {_OUT.relative_to(_ROOT)} 与 {_OUT_JSON.relative_to(_ROOT)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
