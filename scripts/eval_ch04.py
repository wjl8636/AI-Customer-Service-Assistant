"""四策略(vector/bm25/hybrid/hybrid_rerank)分桶评估。需 Milvus Standalone + 已建库 + 上游可调通。

三段指标:
  1) 检索段(确定性):四策略 × 各桶 Recall@5 / MRR(按 expect_section 命中 section_path)。
     召回率按前五截断(RECALL_K),检索深度仍是 Top-10(K),两者分开配。
     跨文档题(E_multi)用 expect_sections_all 写「必须都命中」的几组证据:
     Recall 按组给部分分(凑齐才 1.00),MRR 按每块证据各自名次的倒数取平均、漏的记 0。
  2) 证据覆盖度(确定性·跨策略):四策略召回的 Top-K 证据里,盖住了几个标准答案要点(expect_points 机械匹配)。
  3) 生成段(glm):
     - 四策略各自生成答案 → LLM 判「答案覆盖度」(端到端证明:检索越好→答案越全);
     - hybrid_rerank 额外算 Faithfulness(答案不编造);
     - D 桶经线上管线看拒答率。
生成段带每调用超时 + 失败跳过 + 低并发,单点故障不拖垮整轮;检索段/证据覆盖度不依赖 glm、可复现。

加 --skip-gen 只跑前两段(确定性、不碰裁判模型):裁判上游挂了或只想快速看检索侧时用,
产物照落,页面按「生成段未完成」显示。入口 make eval-rag SKIP_GEN=1。

跑完落两份产物到 data/ch04/reports/:rag_eval.txt 给人读(就是这份运行日志),
rag_eval.json 给后台的 RAG 评估页读。页面只读这份 json、不重算——页面上的数与
终端 make eval-rag 跑出来的必须是同一份。
运行:make eval-rag,或在 /rag-eval 页面上按「重跑 RAG 评估」
"""
import asyncio
import json
import pathlib
import sys
from datetime import datetime

from pydantic import BaseModel, Field

from app.config import settings
from app.core import model_guard, read_notes, retrieval
from app.core import llm
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.kb import milvus_client
from app.tools.builtin import faq

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data/ch04/reports"
_OUT_JSON = _OUT_DIR / "rag_eval.json"
_OUT_TXT = _OUT_DIR / "rag_eval.txt"

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
GRADED_BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi"]  # 可作答桶
# 读图小注要用人话称呼这些策略与题型:喂字段名进去,模型就会照抄 C_colloquial 上页面
STRATEGY_LABEL = {"vector": "纯向量", "bm25": "纯 BM25", "hybrid": "混合",
                  "hybrid_rerank": "混合 + 重排"}
BUCKET_LABEL = {"A_policy": "政策类", "B_model": "型号类",
                "C_colloquial": "口语类", "E_multi": "跨文档类", "overall": "总体"}
SKIP_GEN = "--skip-gen" in sys.argv   # 只跑确定性两段(裁判上游不可用时)
K = 10                 # 检索深度:喂给生成段与证据覆盖度的条数,跟线上 rerank_top_k 对齐
RECALL_K = 5           # 召回率的截断位:一道题需要的证据要在前五条里凑齐。跨文档题一问要
                       # 两三块不同小节的知识,@1 只看第一条根本评不到它们,@5 才评得动
RETR_CONCURRENCY = 5   # 检索:子句拆分后一题要打好几次嵌入+重排,并发压低点少撞限流
GEN_CONCURRENCY = 5    # 生成并发。这一段是整轮的时间大头:300 题四策略约 2200 次裁判调用,
                       # 单次 4~6 秒。压在 5 是给上游留余量(共用 key 的限流是全局的,
                       # 一轮评估把配额吃满,同时在用的其他调用就开始 429)
CALL_TIMEOUT = 45.0    # 单次 glm 调用超时(秒),防卡死

_retr_sem = asyncio.Semaphore(RETR_CONCURRENCY)
_gen_sem = asyncio.Semaphore(GEN_CONCURRENCY)
_errs: list[str] = []


class _Faith(BaseModel):
    faithful: bool = Field(description="是否忠实于证据")
    reason: str = Field(default="")


class _Cov(BaseModel):
    covered_count: int = Field(description="客服答案正确覆盖的要点个数")


COVERAGE_SYS = """你是答案覆盖度评审员。给定用户问题、标准答案要点清单、客服答案。
数一数客服答案里正确覆盖了几个要点:要点信息在答案中有正确体现才算,遗漏、编造或答错都不算。
只需给出覆盖的要点个数(整数),不要超过要点总数。"""
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

COVERAGE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", COVERAGE_SYS),
     ("human", "用户问题:{query}\n\n标准答案要点(共 {n} 个):\n{points}\n\n客服答案:\n{answer}")]
)


_LINES: list[str] = []


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


def _load():
    return [json.loads(ln) for ln in open("tests/data/eval_ch04.jsonl")]


# ---------- 通用 ----------
def _hit_rank(hits, expect_section):
    """这一组证据里任意一个小节命中的最靠前名次;一条都没召回回 0。"""
    for i, h in enumerate(hits):
        sp = h.get("section_path", "") or ""
        if any(w in sp for w in expect_section):
            return i + 1
    return 0


def _groups(sample) -> list[list[str]]:
    """统一成「必须都命中」的组列表:跨文档题一问要好几块知识,写在
    expect_sections_all 里(组内是别名、任一命中即可);单证据题就是一个组。"""
    groups = sample.get("expect_sections_all")
    if groups:
        return [g if isinstance(g, list) else [g] for g in groups]
    return [sample["expect_section"]]


def _ranks_per_group(hits, sample) -> list[int]:
    return [_hit_rank(hits, g) for g in _groups(sample)]


def _recall_at(ranks: list[int], k: int) -> float:
    """Recall@k:这题需要的几块证据,有几成在前 k 条里。
    单证据题就是老口径(0 或 1);跨文档题按组给部分分,凑齐才是 1.00。"""
    return _mean([1.0 if 0 < r <= k else 0.0 for r in ranks])


def _rr(ranks: list[int]) -> float:
    """MRR:单证据题就是命中名次的倒数;跨文档题按「每块证据各自的名次倒数」取平均,
    漏掉的那块记 0。不用「最后一块到位」的名次算,那样两块证据的题上限只有 0.5,
    和单证据桶放进同一张图会被误读成检索变差了。"""
    return _mean([(1.0 / r if r else 0.0) for r in ranks])


def _norm(s: str) -> str:
    return "".join((s or "").split())  # 去所有空白,兼容"每 2 周"↔"每2周"


def _evidence_text(hits) -> str:
    return "\n".join(f"{h.get('question','')}:{h.get('answer','')}" for h in hits)


def _coverage_mech(points, hits) -> float | None:
    """机械证据覆盖度:要点(去空白)是否作为子串出现在召回证据里。"""
    if not points:
        return None
    ev = _norm(_evidence_text(hits))
    return sum(1 for p in points if _norm(p) in ev) / len(points)


def _format_evidence(hits) -> str:
    return "\n".join(f"[{i+1}] {h.get('question','')}:{h.get('answer','')}"
                     for i, h in enumerate(hits))


async def _try(factory, label):
    """带超时地跑一个 glm 协程;失败退避一次再试,仍失败返回 None 并记账(不拖垮整轮)。

    factory 是「造协程的函数」而不是协程本身:协程只能 await 一次,要重试就得重新造一个。
    偶发 429 / 超时直接记成 None 的话,一个桶可能一条样本都不剩,分数就成了 0.00——
    那是「没评上」不是「真的零分」,所以先退一步重试。"""
    for i in range(2):
        try:
            return await asyncio.wait_for(factory(), CALL_TIMEOUT)
        except Exception as e:
            if i == 0:
                await asyncio.sleep(2.0)
                continue
            _errs.append(f"{label}: {type(e).__name__}")
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


# ---------- 段1+2:检索 Recall/MRR + 证据覆盖度(确定性) ----------
async def _retrieve(strat, s):
    async with _retr_sem:
        return await retrieval.search_knowledge(s["query"], strategy=strat, top_k=K)


async def _deterministic(samples):
    """返回 (retrieval_dict, coverage_dict, HITS)。HITS[strat][id]=hits 供生成段复用。"""
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    retrieval_out, coverage_out, HITS = {}, {}, {}
    _log("=== 段1 检索:四策略 × 分桶 Recall@%d / MRR ===" % RECALL_K)
    header = f"{'strategy':16s}" + "".join(f"{b.split('_')[0]:>16s}" for b in GRADED_BUCKETS) + f"{'OVERALL':>16s}"
    _log(header)
    for strat in STRATEGIES:
        hits_list = await asyncio.gather(*(_retrieve(strat, s) for s in graded))
        HITS[strat] = {s["id"]: h for s, h in zip(graded, hits_list)}
        by_bucket, cov_bucket = {}, {}
        row = f"{strat:16s}"
        all_rec, all_rr, all_cov = [], [], []
        for b in GRADED_BUCKETS:
            idx = [i for i, s in enumerate(graded) if s["bucket"] == b]
            ranks = [_ranks_per_group(hits_list[i], graded[i]) for i in idx]
            covs = [_coverage_mech(graded[i]["expect_points"], hits_list[i]) for i in idx]
            rec = _mean([_recall_at(r, RECALL_K) for r in ranks])
            mrr = _mean([_rr(r) for r in ranks])
            by_bucket[b] = {"recall": round(rec, 3), "mrr": round(mrr, 3)}
            cov_bucket[b] = round(_mean(covs), 3)
            all_rec += [_recall_at(r, RECALL_K) for r in ranks]
            all_rr += [_rr(r) for r in ranks]
            all_cov += [c for c in covs if c is not None]
            row += f"  R{rec:.2f}/M{mrr:.2f}".rjust(16)
        by_bucket["overall"] = {"recall": round(_mean(all_rec), 3), "mrr": round(_mean(all_rr), 3)}
        cov_bucket["overall"] = round(_mean(all_cov), 3)
        retrieval_out[strat] = by_bucket
        coverage_out[strat] = cov_bucket
        row += f"  R{_mean(all_rec):.2f}/M{_mean(all_rr):.2f}".rjust(16)
        _log(row)

    _log("\n=== 段2 证据覆盖度(Top-%d 证据盖住标准要点的比例,机械匹配·确定性)===" % K)
    header = f"{'strategy':16s}" + "".join(f"{b.split('_')[0]:>12s}" for b in GRADED_BUCKETS) + f"{'OVERALL':>12s}"
    _log(header)
    for strat in STRATEGIES:
        row = f"{strat:16s}" + "".join(f"{coverage_out[strat][b]:>12.2f}" for b in GRADED_BUCKETS)
        row += f"{coverage_out[strat]['overall']:>12.2f}"
        _log(row)
    return retrieval_out, coverage_out, HITS


# ---------- 段3:四策略生成 + 覆盖度判分 + Faithfulness(glm) ----------
async def _gen_one(strat, s, HITS, model, cov_judge, faith_judge,
                   cov_prompt, faith_prompt):
    hits = HITS[strat][s["id"]]
    evidence = _format_evidence(hits)
    async with _gen_sem:
        ans = await _try(
            lambda: (RAG_ANSWER_PROMPT | model).ainvoke(
                {"query": s["query"], "evidence": evidence}),
            f"gen:{strat}:{s['id']}")
        if ans is None:
            return {"strat": strat, "bucket": s["bucket"], "coverage": None,
                    "faithful": None, "guard_hits": []}
        text = ans.content if isinstance(ans.content, str) else str(ans.content)
        # 型号机械闸:答案里的型号必须逐字来自本轮证据。命中就带着「哪几个没依据」重写一次,
        # 重写还不干净就把它记下来(报告里能看见闸开了几次、拦没拦住),不静悄悄放过去。
        guard_hits = []
        bad = model_guard.unsupported_models(text, evidence)
        if bad:
            guard_hits.append({"id": s["id"], "strat": strat, "models": bad, "fixed": False})
            retry = await _try(
                lambda: (RAG_ANSWER_PROMPT | model).ainvoke(
                    {"query": s["query"],
                     "evidence": evidence + "\n\n" + model_guard.repair_hint(bad)}),
                f"guard-retry:{strat}:{s['id']}")
            if retry is not None:
                fixed_text = retry.content if isinstance(retry.content, str) else str(retry.content)
                if not model_guard.unsupported_models(fixed_text, evidence):
                    text = fixed_text
                    guard_hits[-1]["fixed"] = True
        cov = await _try(
            lambda: (cov_prompt | cov_judge).ainvoke(
                {"query": s["query"], "n": len(s["expect_points"]),
                 "points": "\n".join(f"{i+1}. {p}" for i, p in enumerate(s["expect_points"])),
                 "answer": text}),
            f"cov:{strat}:{s['id']}")
        coverage = None
        if cov is not None and s["expect_points"]:
            coverage = min(cov.covered_count, len(s["expect_points"])) / len(s["expect_points"])
        faithful = None
        faith_detail = None
        if strat == "hybrid_rerank":
            fa = await _try(
                lambda: (faith_prompt | faith_judge).ainvoke(
                    {"evidence": evidence, "answer": text}),
                f"faith:{s['id']}")
            if fa is not None:
                faithful = bool(fa.faithful)
                if not faithful:  # 不忠实个案:留证据供报告展示(问题/答案/裁判理由/角标原文)
                    # citations 存这一轮喂给模型的 Top-K 证据全集,顺序与 evidence 拼接一致——
                    # 答案里的 [n] 就是这份列表的序号(答案通常只引其中两三条)。裁判说「证据里没有」,
                    # 追溯时得能看到手里到底有哪几块、又只用了哪几块
                    faith_detail = {"id": s["id"], "bucket": s["bucket"],
                                    "query": s["query"], "answer": text, "reason": fa.reason,
                                    "citations": [
                                        {"n": i + 1, "chunk_id": h.get("id"),
                                         "section_path": h.get("section_path", ""),
                                         "question": h.get("question", ""),
                                         "answer": h.get("answer", "")}
                                        for i, h in enumerate(hits)]}
    return {"strat": strat, "bucket": s["bucket"], "coverage": coverage,
            "faithful": faithful, "faith_detail": faith_detail,
            "guard_hits": guard_hits}


async def _ledger(cases) -> None:
    """把这一轮的编造个案写进台账(faith_cases)。

    报告一重跑就被覆盖,个案和它的处置状态得有地方长期待着——同一道题跨轮反复被判编造,
    才看得出库里那一格一直没补对。写库失败不影响这一轮的报告:台账是附加的管理视图,
    不能让它把评估结果拖没了(比如没应用 sql/ch04-ddl.sql 的环境)。
    """
    if not cases:
        _log("\n-- 编造个案台账:本轮 0 例,无需写入 --")
        return
    try:
        from app.db import repository
        new = again = reopened = 0
        for c in cases:
            _, was_resolved = await repository.upsert_faith_case(
                c["id"], c["bucket"], c["query"], c["answer"], c["reason"],
                citations=c.get("citations"), judge_model=settings.chat_model)
            if was_resolved:
                reopened += 1
        rows, total, counts = await repository.list_faith_cases(None, 1, 1)
        _log(f"\n-- 编造个案台账:本轮 {len(cases)} 例已写入,台账共 {total} 条"
             f"(未解决 {counts['未解决']} · 已解决 {counts['已解决']} · 无需解决 {counts['无需解决']})"
             + (f";其中 {reopened} 条是处置过又复发" if reopened else "") + " --")
    except Exception as e:
        _log(f"\n-- 编造个案台账写入失败({type(e).__name__}),报告不受影响;"
             f"检查是否已应用 sql/ch04-ddl.sql --")


async def _refusal_one(s):
    """D 桶一题:走线上管线看该拒有没有拒。没拒住的把证据留下来——
    只报一个 56/57 的比率,读者没法知道是哪一条漏了、被什么证据骗过去的。"""
    async with _gen_sem:
        out = await _try(lambda: faq.query_faq.ainvoke({"keyword": s["query"]}),
                         f"refuse:{s['id']}")
    if out is None:
        return None
    refused = not out.get("sufficient")
    detail = None
    if not refused:
        cites = out.get("citations") or []
        top = cites[0] if cites else {}
        detail = {"id": s["id"], "query": s["query"],
                  "section_path": top.get("section_path", ""),
                  "evidence": f"{top.get('question', '')}:{top.get('answer', '')}"[:220]}
    return {"refused": refused, "detail": detail}


async def _generation(samples, HITS):
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    absent = [s for s in samples if s["bucket"] == "D_absent"]
    model = get_chat_model()
    # 结构化输出统一走 app.core.llm.structured:只走 function calling,失败重试一次就抛
    # 两个裁判都跑 temperature=0:同一份证据+答案重放,0.3 下的判断会来回跳(见 judge_check),
    # 那点抖动直接写进分数里。答用户的话可以有温度,评分不能有
    cov_judge = llm.structured(_Cov, temperature=0)
    faith_judge = llm.structured(_Faith, temperature=0)
    cov_prompt, faith_prompt = COVERAGE_PROMPT, FAITHFULNESS_PROMPT

    _log("\n=== 段3 生成:四策略答案覆盖度(glm 判分)+ Faithfulness + D 桶拒答 ===")
    tasks = [_gen_one(strat, s, HITS, model, cov_judge, faith_judge, cov_prompt, faith_prompt)
             for strat in STRATEGIES for s in graded]
    tasks += [_refusal_one(s) for s in absent]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    n_gen = len(STRATEGIES) * len(graded)
    gen_results = [r for r in results[:n_gen] if isinstance(r, dict)]
    raw_refusals = results[n_gen:]
    refusals = [r for r in raw_refusals if isinstance(r, dict) and "refused" in r]
    # 没评上的那几条也点名:57/60 里缺的是哪三题,读者不该靠猜
    refusal_skipped = [s["id"] for s, r in zip(absent, raw_refusals)
                       if not (isinstance(r, dict) and "refused" in r)]

    # 上游整段挂掉(余额耗尽、鉴权失效)时一条都判不出来。这时按「生成段未完成」处理,
    # 不能把一片 0.00 落进产物和页面——那会被读成「覆盖度真的是零」。
    if not refusals and not any(r["coverage"] is not None or r["faithful"] is not None
                                for r in gen_results):
        raise RuntimeError(f"生成段 {len(_errs)} 次调用全部失败(上游不可用),不落零分")

    # 四策略 × 分桶 答案覆盖度
    answer_cov = {}
    _log("\n-- 答案覆盖度(生成答案盖住标准要点的比例)--")
    header = f"{'strategy':16s}" + "".join(f"{b.split('_')[0]:>12s}" for b in GRADED_BUCKETS) + f"{'OVERALL':>12s}"
    _log(header)
    for strat in STRATEGIES:
        by = {}
        allc = []
        for b in GRADED_BUCKETS:
            cs = [r["coverage"] for r in gen_results if r["strat"] == strat and r["bucket"] == b and r["coverage"] is not None]
            # 一条都没评上就记 None:0.00 会被读成「真的零分」,而这是「没评上」
            by[b] = round(_mean(cs), 3) if cs else None
            allc += cs
        by["overall"] = round(_mean(allc), 3) if allc else None
        answer_cov[strat] = by
        fmt = lambda v: ("     未评上" if v is None else f"{v:>12.2f}")  # noqa: E731
        _log(f"{strat:16s}" + "".join(fmt(by[b]) for b in GRADED_BUCKETS) + fmt(by["overall"]))

    # Faithfulness(hybrid_rerank)分桶
    faithfulness = {}
    _log("\n-- Faithfulness(hybrid_rerank 线上管线,答案不编造)--")
    for b in GRADED_BUCKETS:
        fs = [r["faithful"] for r in gen_results if r["strat"] == "hybrid_rerank" and r["bucket"] == b and r["faithful"] is not None]
        faithfulness[b] = {"v": round(_mean(fs), 3) if fs else None, "answered": len(fs)}
        shown = "未评上" if not fs else f"{faithfulness[b]['v']:.3f}"
        _log(f"{b:14s} faithfulness={shown} (评 {len(fs)} 题)")

    # 不忠实(编造)个案:问题/答案/裁判理由,供报告 §03 展开查看
    faith_cases = [r["faith_detail"] for r in gen_results if r.get("faith_detail")]
    faith_cases.sort(key=lambda c: c["id"])
    if faith_cases:
        _log(f"\n-- 被判不忠实(编造)个案:{len(faith_cases)} 例 → {', '.join(c['id'] for c in faith_cases)} --")
    else:
        _log("\n-- 被判不忠实(编造)个案:0 例 --")

    # D 桶拒答率(线上管线 hybrid_rerank + 证据闸)
    correct = sum(1 for r in refusals if r["refused"])
    # 一条都没评上就记 None:0.000 会被读成「全都没拒住」,而这是「没评上」
    rate = (correct / len(refusals)) if refusals else None
    shown_rate = "未评上" if rate is None else f"{rate:.3f}"
    _log(f"\n-- D 桶拒答率(线上管线)= {shown_rate}({correct}/{len(refusals)} 正确拒答;评 {len(refusals)}/{len(absent)} 题)--")
    miss_cases = [r["detail"] for r in refusals if r["detail"]]
    miss_cases.sort(key=lambda c: c["id"])
    if miss_cases:
        for c in miss_cases:
            _log(f"   该拒没拒:{c['id']} 「{c['query']}」← 被当成答案的证据:{c['section_path']}")
    else:
        _log("   该拒没拒:0 例")
    if refusal_skipped:
        _log(f"   未评上(调用失败/超时):{', '.join(refusal_skipped)}")

    # 型号机械闸的战绩:开了几次、重写救回几次、还剩几条带着没依据的型号出去
    guard_hits = [h for r in gen_results for h in r.get("guard_hits", [])]
    guard = {"hits": len(guard_hits), "fixed": sum(1 for h in guard_hits if h["fixed"]),
             "cases": sorted(guard_hits, key=lambda h: (h["id"], h["strat"]))}
    _log(f"\n-- 型号机械闸:命中 {guard['hits']} 次,重写救回 {guard['fixed']} 次 --")
    for h in guard["cases"]:
        _log(f"   {h['id']}({h['strat']}) 证据里没有的型号:{'、'.join(h['models'])}"
             f" → {'重写后干净' if h['fixed'] else '重写后仍不干净'}")

    await _ledger(faith_cases)

    return {"answer_coverage": answer_cov, "faithfulness": faithfulness,
            "faithfulness_cases": faith_cases, "model_guard": guard,
            "refusal": {"rate": (round(rate, 3) if rate is not None else None),
                        "total": len(refusals), "correct": correct,
                        "cases": miss_cases, "skipped_ids": refusal_skipped},
            "skipped": len(_errs)}


# ---------- 产物 ----------
async def _kb_chunks():
    try:
        def work():
            c = milvus_client.get_client()
            milvus_client.ensure_collection(c)
            return milvus_client.count(c)
        return await milvus_client.acall(work)
    except Exception:
        return "—"


def _labeled(data, pick=None):
    """把 {策略键:{桶键:值}} 换成中文标签的版本,专供读图小注的 payload。"""
    return {STRATEGY_LABEL[s]: {BUCKET_LABEL[b]: (pick((data[s].get(b) or {})) if pick
                                                 else (data[s] or {}).get(b))
                                for b in GRADED_BUCKETS + ["overall"]}
            for s in STRATEGIES}


async def _read_notes(retrieval_data, coverage_data, generation_data):
    """四张图各配一句读图小注,和报告一起落盘;拿不到就空着,页面回落自己那句兜底话。"""
    jobs = {
        "rag_mrr": _labeled(retrieval_data, lambda d: d.get("mrr")),
        "rag_recall": _labeled(retrieval_data, lambda d: d.get("recall")),
        "rag_coverage": _labeled(coverage_data),
    }
    if generation_data:
        jobs["rag_answer_coverage"] = _labeled(generation_data["answer_coverage"])
    notes = await read_notes.generate_all(jobs)
    _log(f"\n读图小注:生成 {len(notes)}/{len(jobs)} 条"
         + ("" if len(notes) == len(jobs) else "(缺的那几张页面用兜底句)"))
    return notes


def _write_report(retrieval_data, coverage_data, generation_data, meta, notes=None):
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_TXT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    report = {"meta": meta, "retrieval": retrieval_data,
              "evidence_coverage": coverage_data, "generation": generation_data,
              "read_notes": notes or {}}
    # generation 为 None 也照样落盘:页面据此显示「生成段未完成」,而不是一片空白
    _OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")


async def main():
    samples = _load()
    import collections as _c
    per = _c.Counter(s["bucket"] for s in samples)
    _log(f"评估集:{len(samples)} 题(" + "、".join(
        f"{BUCKET_LABEL.get(b, b)} {n}" for b, n in per.most_common()) + f"),策略 {STRATEGIES}")
    _log(f"检索并发 {RETR_CONCURRENCY} · 生成并发 {GEN_CONCURRENCY} · 单调用超时 {CALL_TIMEOUT:.0f}s\n")
    retrieval_data, coverage_data, HITS = await _deterministic(samples)
    generation_data = None
    if SKIP_GEN:
        _log("\n[生成段跳过] --skip-gen:本轮只跑确定性两段,页面按「生成段未完成」显示。")
    else:
        try:
            generation_data = await _generation(samples, HITS)
        except Exception as e:
            _log(f"\n[生成段未完成] {type(e).__name__}: {str(e)[:140]}")
            _log("检索段与证据覆盖度已完成;页面将标注生成段未完成,上游恢复后重跑补全。")
    if _errs:
        _log(f"\n注:生成段有 {len(_errs)} 次 glm 调用超时/失败被跳过(上游不稳),已按可用样本计。")
    meta = {"n_samples": len(samples), "kb_chunks": await _kb_chunks(),
            "recall_k": RECALL_K, "retrieve_k": K,
            "concurrency": GEN_CONCURRENCY, "chat_model": settings.chat_model,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    # 读图小注也要过裁判模型,--skip-gen 时一并跳过,页面回落自己那句兜底话
    notes = {} if SKIP_GEN else await _read_notes(retrieval_data, coverage_data,
                                                  generation_data)
    _write_report(retrieval_data, coverage_data, generation_data, meta, notes)
    _log(f"\n已生成:{_OUT_JSON.relative_to(_ROOT)} · {_OUT_TXT.relative_to(_ROOT)}")
    _log("页面:http://localhost:8000/rag-eval(后台 → RAG 评估)")
    _log("完成。" if generation_data is not None else "完成(仅检索段+证据覆盖度;生成段待上游恢复重跑)。")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
