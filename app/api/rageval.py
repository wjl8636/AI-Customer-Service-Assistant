# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""RAG 评估页的只读 API:把 make eval-rag 落下的那份报告端出去,不重算。

页面上的每个数都来自 data/ch04/reports/rag_eval.json——那是评估脚本跑完写下的产物。
API 不复算 MRR、不复判覆盖度:一旦这里也算一遍,页面和终端就会有两个真相,谁对都说不清。

产物没跑过不是错误:回 present=false + 该按哪个作业,页面据此长出「去跑一次」而不是白屏。
生成段(LLM 裁判那一段)可能因上游不稳而缺,报告里 generation 为 null,页面照样呈现
确定性的检索段——半份结果比一片空白有用。
"""
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query

from app.core import jobs
from app.db import repository
from app.schemas.rageval import FaithCaseStatusRequest

router = APIRouter(prefix="/api/rag-eval")

REPORT = pathlib.Path(__file__).resolve().parents[2] / "data/ch04/reports/rag_eval.json"
TEXT_LOG = REPORT.with_name("rag_eval.txt")
JOB = "eval-rag"          # 重跑按钮按的就是终端那条 make eval-rag
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")
BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")   # 可作答桶;库外 D_absent 只评拒答


def _read() -> dict | None:
    try:
        return json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stat(path: pathlib.Path) -> dict:
    try:
        st = path.stat()
    except OSError:
        return {"present": False, "bytes": None, "mtime": None}
    import datetime as dt
    return {"present": True, "bytes": st.st_size,
            "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}


def _best_strategy(report: dict) -> dict:
    """总体 MRR 最高的那一路。四策略对照的结论句、KPI 都指着它,算一次就够。"""
    best, best_mrr = None, -1.0
    for s in STRATEGIES:
        mrr = ((report.get("retrieval") or {}).get(s) or {}).get("overall", {}).get("mrr")
        if mrr is not None and mrr > best_mrr:
            best, best_mrr = s, mrr
    return {"strategy": best, "mrr": best_mrr if best else None}


@router.get("/overview")
async def overview() -> dict:
    report = _read()
    job = {"specs": [jobs.status(JOB)], "artifacts": {
        "json": _stat(REPORT) | {"path": "data/ch04/reports/rag_eval.json"},
        "text": _stat(TEXT_LOG) | {"path": "data/ch04/reports/rag_eval.txt"},
    }}
    if report is None:
        return {"present": False, "job": job,
                "make": "make eval-rag",
                "hint": "还没跑过 RAG 评估。按「重跑 RAG 评估」现场跑一轮"
                        "(四策略 × 四桶,需 Milvus + 已建库 + 聊天上游,分钟级)。"}
    gen = report.get("generation")
    return {
        "present": True,
        "meta": report.get("meta") or {},
        "retrieval": report.get("retrieval") or {},
        "evidence_coverage": report.get("evidence_coverage") or {},
        "generation": gen,
        "generation_done": gen is not None,
        # 读图小注:评估脚本落盘时让模型看着这一轮的数写的,并已校过里面每个数字。
        # 缺哪张就没哪个 key,页面回落自己那句兜底话——注不是数,页面不能因为没注就白屏
        "read_notes": report.get("read_notes") or {},
        "best": _best_strategy(report),
        "strategies": list(STRATEGIES),
        "buckets": list(BUCKETS),
        "job": job,
    }


# ---------- 编造个案台账(跨轮累计,不随报告覆盖而消失) ----------
def _case_out(r) -> dict:
    return {
        "id": r.id, "eval_id": r.eval_id, "bucket": r.bucket, "query": r.query,
        "strategy": r.strategy, "answer": r.answer, "reason": r.reason,
        # 角标原文:答案里的 [n] 对应哪块证据。老数据没记为 null,页面据此提示「这轮没留快照」
        "citations": r.citations, "judge_model": r.judge_model,
        "resolution": r.resolution,
        "status": r.status, "seen_count": r.seen_count,
        # 处置过又被判出来 = 复发,列表上要打标:不是新问题,是上次没改对
        "reopened": bool(r.resolved_at) and r.status == "未解决",
        "first_seen_at": r.first_seen_at.isoformat(timespec="seconds") if r.first_seen_at else None,
        "last_seen_at": r.last_seen_at.isoformat(timespec="seconds") if r.last_seen_at else None,
        "resolved_at": r.resolved_at.isoformat(timespec="seconds") if r.resolved_at else None,
    }


def _hallucination(counts: dict, report: dict | None, status_map: dict) -> dict:
    """幻觉率两个口径,页面上并排显示、别混着看。

    **两个口径都算「本轮」**:分子只数这一轮报告判出的那几题(`faithfulness_cases`),
    分母是这一轮评上的题数。台账是跨轮累计的,拿它的总条数当分子、再除以一轮的题量,
    等于把历史上所有轮次的个案都摊到这一轮头上——上一轮个案恰好等于台账全部时看不出来,
    修完几条之后就会虚高。台账累计另开一行显示(`ledger`),两件事别混。

    **两类幻觉都算进去**:
    1. 可作答题上「答了但编了」——忠实度裁判判出的编造个案;
    2. 库外题上「本该拒答却答了」——该拒没拒。库里根本没有这个知识还答出来,那就是无据而答,
       和编造是一回事,所以分母要含库外桶、分子要含这些条。

    - **裁判判出率** = (本轮判出的编造个案 + 该拒没拒) / 参与评估的题数。这是**线索量**,编造
      那部分里混着裁判判严的。
    - **确认幻觉率** = (本轮判出的个案里被人工点成「已解决」的 + 该拒没拒) / 参与评估的题数。
      编造那部分只算人工过目、确认真编了并且改掉的:「无需解决」是裁判判严了(不算),「未解决」
      是还没过目(先不算,所以这个数是下界)。**该拒没拒不用人工确认**——库外题只要答了就是
      无据而答;要是发现那题其实库里有答案(评估集标错了),该改的是评估集,改完这条自然就不在了。

    分母用报告里「实际评上的题数」而不是评估集总题数:调用失败没评上的题不该摊进分母。
    没跑过报告就没有分母,回 None——不知道的时候不编一个出来。
    """
    gen = (report or {}).get("generation") or {}
    faith = gen.get("faithfulness") or {}
    graded = sum((v or {}).get("answered") or 0 for v in faith.values())
    refusal = gen.get("refusal") or {}
    absent = refusal.get("total") or 0
    missed = max(0, absent - (refusal.get("correct") or 0))     # 该拒没拒 = 无据而答
    evaluated = (graded + absent) or None

    # 本轮判出的是哪几题由报告说;每题现在什么处置由台账说
    round_ids = [c.get("id") for c in (gen.get("faithfulness_cases") or []) if c.get("id")]
    tally = {"未解决": 0, "已解决": 0, "无需解决": 0}
    for eid in round_ids:
        st = status_map.get(eid)
        if st in tally:
            tally[st] += 1
    cases_judged, confirmed_cases = len(round_ids), tally["已解决"]
    rate = (lambda n: round(n / evaluated, 4) if evaluated else None)
    return {"evaluated": evaluated, "graded": graded or None, "absent": absent or None,
            "cases_judged": cases_judged, "cases_confirmed": confirmed_cases,
            "pending": tally["未解决"], "dismissed": tally["无需解决"],
            "refusal_missed": missed,
            "judged": cases_judged + missed, "confirmed": confirmed_cases + missed,
            "judged_rate": rate(cases_judged + missed),
            "confirmed_rate": rate(confirmed_cases + missed),
            # 台账累计:跨轮管理视图,和上面两个「本轮」口径不是一回事
            "ledger": {"total": sum(counts.values()), **counts}}


@router.get("/faith-cases")
async def faith_cases(status: str | None = None,
                      page: int = Query(1, ge=1),
                      size: int = Query(5, ge=1, le=50)) -> dict:
    """分页台账 + 幻觉率。status 传空 = 全部;未解决排最前面(该处理的先看见)。"""
    if status is not None and status not in ("未解决", "已解决", "无需解决"):
        raise HTTPException(status_code=400, detail="status 只能是 未解决/已解决/无需解决")
    rows, total, counts = await repository.list_faith_cases(status, page, size)
    status_map = await repository.faith_case_status_map()
    return {"items": [_case_out(r) for r in rows], "total": total, "page": page,
            "size": size, "pages": max(1, -(-total // size)), "counts": counts,
            "hallucination": _hallucination(counts, _read(), status_map)}


@router.post("/faith-cases/{case_id}/status")
async def set_faith_case_status(case_id: int, req: FaithCaseStatusRequest) -> dict:
    """处置必须留一句交代:没有理由的「无需解决」等于没处理过,半年后没人说得清。"""
    note = (req.resolution or "").strip()
    if req.status != "未解决" and not note:
        raise HTTPException(
            status_code=400,
            detail="标「已解决」要写清怎么解决的,标「无需解决」要写清为什么不用改")
    row = await repository.set_faith_case_status(case_id, req.status, note or None)
    if row is None:
        raise HTTPException(status_code=404, detail="个案不存在")
    return _case_out(row)
