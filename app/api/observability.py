"""「观测与成本」页的只读 API:ch09 那三张终端报表搬进后台,一页看齐、也能在页上重跑。

三块各有自己的权威源,这里只负责端出去:

- 意图成本账:`data/ch09/reports/cost_by_intent.json`(make cost-report 落的产物),
  平均 token 与占比都是脚本算好写进去的,这里不复算——不然页面和终端会有两个数。
- 评估趋势:`eval_runs` 表就是趋势本身的权威源(make eval-flywheel 每轮插一行),
  所以直接读表,不去读那份文本报告。涨跌箭头由页面按同一批行算,不是另一份数据。
- 置信度阈值校准:`data/ch09/reports/confidence_calibration.json`(make calibrate-confidence),
  另带一份「当前在用的阈值」,让人看得出推荐值和线上值对不对得上。

每块的「读图」小注也来自产物(脚本落盘时让模型看着那一轮的数写的),这里只端出去。
趋势那句注旁挂在 eval_trend_note.json 里,只有它记的轮次还是最新一轮时才端,
不然就是拿上一轮的话解释这一轮的数。

三块互不连坐:Langfuse 没起只是成本账没产物,mysql 挂了只是趋势读不到,
剩下的照样出——依赖不齐的时候恰恰最该看这一页。
"""
import json
import pathlib

from fastapi import APIRouter

from app.config import settings
from app.core import jobs
from app.core.confidence import W_KEY, W_MARGIN, W_TOP1, W_VALID
from app.db import repository

router = APIRouter(prefix="/api/observability")

REPORT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data/ch09/reports"
COST = REPORT_DIR / "cost_by_intent.json"
CALIB = REPORT_DIR / "confidence_calibration.json"
TREND_NOTE = REPORT_DIR / "eval_trend_note.json"
COST_JOB, TREND_JOB, CALIB_JOB = "cost-report", "eval-flywheel", "calibrate-confidence"
TREND_LIMIT = 10                  # 趋势看最近十轮:再多一屏也读不出走向
# recall_at_10 是改口径之前的老键,历史那几轮还在库里,一起列出来页面才画得全
METRIC_NAMES = ("recall_at_5", "recall_at_10", "mrr", "faithfulness", "refusal_rate")


def _read(path: pathlib.Path) -> dict | None:
    """产物读不出来(没跑过、或写坏了半个 json)一律当没有,让页面提示重跑,不抛 500。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _block(job: str, make: str, hint: str) -> dict:
    return {"present": False, "status": "missing", "job": jobs.status(job),
            "make": make, "hint": hint}


def _cost_block() -> dict:
    block = _block(COST_JOB, "make cost-report",
                   "还没跑过意图成本账。先在聊天页问几句攒 trace,再按「重跑 意图成本账」。")
    report = _read(COST)
    if report is None:
        return block
    rows = report.get("rows") or []
    return block | {
        "present": True, "status": "ok" if rows else "missing",
        "meta": report.get("meta") or {},
        "rows": rows,
        "total_tokens": report.get("total_tokens"),
        "total_requests": report.get("total_requests"),
        # 最烧钱那一路:脚本按 tokens 降序写的,取第一行就是,不重排
        "top": rows[0] if rows else None,
        "read_note": (report.get("read_notes") or {}).get("cost_by_intent"),
        "hint": None if rows else "窗口内没有带 intent tag 的 trace,先在聊天页问几句再重跑。",
    }


def _trend_note(latest_run_id: int) -> str | None:
    """趋势那句注只在它描述的轮次还是最新一轮时才作数,过期的宁可不显示。"""
    note = _read(TREND_NOTE) or {}
    return note.get("note") if note.get("run_id") == latest_run_id else None


async def _trend_block() -> dict:
    block = _block(TREND_JOB, "make eval-flywheel",
                   "还没跑过评估流水线。按一次「重跑 评估流水线」,这一轮就是趋势的第一个点。")
    try:
        runs = await repository.list_eval_runs(limit=TREND_LIMIT)
    except Exception as e:
        return block | {"status": "error", "note": f"{type(e).__name__}: {e}"}
    return block | {
        "present": bool(runs), "status": "ok" if runs else "missing",
        "metric_names": list(METRIC_NAMES),
        "read_note": _trend_note(runs[0].id) if runs else None,
        # 新在上:第 0 行是最近一轮,页面据此和下一行比涨跌
        "runs": [{"id": r.id, "triggered_by": r.triggered_by,
                  "dataset_size": r.dataset_size, "metrics": r.metrics or {},
                  "created_at": r.created_at.isoformat(timespec="seconds") if r.created_at else None}
                 for r in runs],
    }


def _calibration_block() -> dict:
    block = _block(CALIB_JOB, "make calibrate-confidence",
                   "还没校准过。按「重跑 置信度阈值校准」在 ch04 评估集上扫一遍,阈值就不用拍脑袋。")
    block["in_use"] = settings.evidence_confidence_threshold
    block["weights"] = {"top1": W_TOP1, "valid_count": W_VALID,
                        "margin": W_MARGIN, "key_clause": W_KEY}
    report = _read(CALIB)
    if report is None:
        return block
    rec = report.get("recommended") or {}
    scan = report.get("scan") or []
    in_use = settings.evidence_confidence_threshold
    rec_t = rec.get("threshold")
    # 在 scan 里查在用阈值对应的通过率/放行率
    in_use_stats = next((s for s in scan if s["t"] == round(in_use, 2)), None)
    if rec_t == in_use:
        sync = "match"
    elif in_use > rec_t:
        sync = "conservative"  # 故意选高于推荐值:更保守,误放更低
    else:
        sync = "aggressive"    # 低于推荐值:应拒可能漏进来
    return block | {
        "present": True, "status": "ok",
        "meta": report.get("meta") or {},
        "distribution": report.get("distribution") or {},
        "scan": scan,
        "recommended": rec,
        "in_use_stats": in_use_stats,
        "read_note": (report.get("read_notes") or {}).get("confidence_calibration"),
        "in_sync": sync,
    }


@router.get("/overview")
async def overview() -> dict:
    return {"cost": _cost_block(), "trend": await _trend_block(),
            "calibration": _calibration_block()}
