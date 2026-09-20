# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""知识库录入 API:把建库这条链搬上页面——材料清单、切块预览、录入入库、向量化、检索自测。

两条录入路径,边界分清:
  1. 手工录入(POST /preview → /ingest):正文在浏览器里贴,预览什么样就入库什么样。
     切块用的是 app.kb.documents.build_chunks,与离线 CLI 同一个函数,不在接口里另写一套。
  2. 离线建库(作业按钮):data/kb/*.md 四份材料走 make kb-build / kb-mine / kb-vectorize,
     配方是仓库里那一份,页面按的与终端敲的是同一条命令。

查重按「问法 + 正文」的指纹,不是只按问法:表格按行拆、超长散文递归切,同一节切出的多块
共用节标题,只按问法查会把它们当重复误杀。同一份正文重复录入则天然幂等——指纹全撞,全跳过。
"""
import datetime as dt
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core import jobs, retrieval
from app.db import repository
from app.kb import chunking, dedup, documents, dualwrite, milvus_client
from app.kb.sources import CONTENT_TYPE_DESC, CONTENT_TYPES, KB_DIR, SOURCE_TYPES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb")

MAX_TEXT_CHARS = 40000          # 单次录入正文上限:一份章节文档够用,防误贴整本手册把上游打爆
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")
# 页面上这一排按钮对应的作业(注册表在 app/core/jobs.py,前端只传名字)
KB_JOBS = ("kb-preview", "kb-build", "kb-repatch", "kb-mine", "kb-vectorize",
           "seed-conv", "kb-reset")


def _fingerprint(questions: str, answer: str) -> str:
    """查重指纹:问法与正文各自去掉空白标点后拼一起。"""
    return dedup.normalize_question(questions) + "|" + dedup.normalize_question(answer)


def _chunk_view(seq: int, c: documents.Chunk, dup: bool | None) -> dict:
    return {
        "seq": seq,
        "section_path": c.section_path or "(无标题层级)",
        "category": c.category,
        "questions": c.questions,
        "answer": c.answer,
        "chars": len(c.answer),
        "is_key_clause": bool(c.is_key_clause),
        "is_table": chunking.is_table_block(c.answer),
        "duplicate": dup,
    }


def _features(chunks: list[documents.Chunk]) -> dict:
    """把三个切块特性有没有触发直接标出来:按标题层级切 / 表格按行拆 / 句末重叠。
    一节被切成多块才谈得上重叠或拆表——按 section_path 分组数就能看出来。"""
    groups: dict[str, list[documents.Chunk]] = {}
    for c in chunks:
        groups.setdefault(c.section_path, []).append(c)
    table_split = any(len(g) > 1 and chunking.is_table_block(g[0].answer) for g in groups.values())
    overlap = any(len(g) > 1 and not chunking.is_table_block(g[0].answer) for g in groups.values())
    return {
        "sections": len(groups),
        "table_split": table_split,
        "overlap": overlap,
        "multi_piece_sections": [p for p, g in groups.items() if len(g) > 1],
    }


async def _existing_fingerprints() -> set[str] | None:
    """库里已有的指纹集合;库连不上回 None——预览照样能看,只是查重那列显示未知。"""
    try:
        return {_fingerprint(q, a) for q, a in await repository.list_chunk_pairs()}
    except Exception:
        return None


async def milvus_state() -> dict:
    """Milvus 探活 + 条数。连不上不是错误,是一种状态:页面显示离线,双写一致那一栏不下结论。"""
    try:
        def work():
            c = milvus_client.get_client()
            milvus_client.ensure_collection(c)
            return milvus_client.count(c)
        return {"online": True, "count": await milvus_client.acall(work),
                "collection": milvus_client.COLLECTION}
    except Exception as e:
        return {"online": False, "count": None, "detail": f"{type(e).__name__}: {e}"}


def _sources() -> list[dict]:
    """建库材料清单 + 就地切块(dry-run,不写库、不碰 Milvus、不调上游)。"""
    out = []
    for fname, ctype in SOURCE_TYPES.items():
        path = KB_DIR / fname
        item = {"file": fname, "content_type": ctype, "present": path.exists(),
                "path": f"data/kb/{fname}"}
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            chunks = documents.build_chunks(raw, content_type=ctype)
            st = path.stat()
            item.update({
                "chars": len(raw), "lines": raw.count("\n") + 1,
                "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "chunks": len(chunks),
                "key_clause": sum(c.is_key_clause for c in chunks),
                "features": _features(chunks),
            })
        out.append(item)
    return out


@router.get("/overview")
async def overview() -> dict:
    """录入页总览:库存 + 双写一致 + 暂存表 + 材料清单 + 作业状态。"""
    milvus = await milvus_state()
    try:
        stats = await repository.knowledge_stats()
        recent = [{"id": r.id, "questions": r.questions, "answer": r.answer[:120],
                   "category": r.category, "section_path": r.section_path,
                   "content_type": r.content_type, "is_key_clause": bool(r.is_key_clause),
                   "status": r.vectorize_status,
                   "created_at": r.created_at.isoformat(timespec="seconds") if r.created_at else None}
                  for r in await repository.list_recent_chunks(12)]
        staging = await repository.staging_stats()
        db_err = None
    except Exception as e:                     # mysql 没起时不连坐整页:材料清单与作业按钮照样能用
        stats = {"total": None, "pending": None, "done": None,
                 "by_content_type": {}, "key_clause": None}
        recent, staging, db_err = [], None, f"{type(e).__name__}: {e}"

    # 双写一致 = 没有块卡在 pending,且 MySQL 已向量化数与 Milvus 条数对得上。
    # 任一边读不到就是 None(不知道),不当成不一致——「读不到」和「对不上」是两回事。
    consistent = None
    if db_err is None and milvus["online"]:
        consistent = stats["pending"] == 0 and stats["done"] == milvus["count"]

    return {
        "chunks": stats, "recent": recent, "staging": staging, "db_error": db_err,
        "milvus": milvus, "consistent": consistent,
        "sources": _sources(),
        "content_types": [{"key": k, "desc": CONTENT_TYPE_DESC[k]} for k in CONTENT_TYPES],
        "jobs": [jobs.status(n) for n in KB_JOBS],
    }


class PreviewIn(BaseModel):
    text: str | None = None
    file: str | None = None                    # 只能是材料清单里的文件名,不接受路径
    content_type: str = "faq"


def _resolve_preview(body: PreviewIn) -> tuple[str, str, str]:
    """回 (正文, content_type, 来源标注)。file 走白名单精确匹配,传不进任何路径。"""
    if body.file:
        if body.file not in SOURCE_TYPES:
            raise HTTPException(status_code=400, detail=f"不在建库材料清单里:{body.file}")
        path = KB_DIR / body.file
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"材料文件不存在:data/kb/{body.file}")
        return path.read_text(encoding="utf-8"), SOURCE_TYPES[body.file], f"data/kb/{body.file}"
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="正文是空的,贴一段 Markdown 再预览")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"正文 {len(text)} 字,超过单次上限 {MAX_TEXT_CHARS} 字,拆开录")
    if body.content_type not in CONTENT_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"content_type 只能是 {'/'.join(CONTENT_TYPES)}")
    return text, body.content_type, "手工录入"


@router.post("/preview")
async def preview(body: PreviewIn) -> dict:
    """切块预览(dry-run):不写库、不碰 Milvus、不调上游。预览什么样,入库就是什么样。"""
    text, ctype, source = _resolve_preview(body)
    chunks = documents.build_chunks(text, content_type=ctype)
    seen = await _existing_fingerprints()
    views, dups, batch = [], 0, set()
    for i, c in enumerate(chunks, 1):
        fp = _fingerprint(c.questions, c.answer)
        dup = None if seen is None else (fp in seen or fp in batch)
        dups += bool(dup)
        batch.add(fp)
        views.append(_chunk_view(i, c, dup))
    return {"source": source, "content_type": ctype, "chars": len(text),
            "total": len(chunks), "duplicates": dups,
            "key_clause": sum(c.is_key_clause for c in chunks),
            "features": _features(chunks), "chunks": views,
            "dedup_known": seen is not None}


class IngestIn(BaseModel):
    text: str
    content_type: str = "faq"
    vectorize: bool = True          # 入库后顺手把库里所有 pending 补齐(与 kb-vectorize 同一个函数)


@router.post("/ingest")
async def ingest(body: IngestIn) -> dict:
    """录入:切块 → 按指纹查重 → 写 MySQL(pending)→ 可选立刻向量化。

    先写 MySQL 再进 Milvus 这个顺序不能反:MySQL 是原文权威源,先落地再向量化,
    中间挂了重跑捡 pending 就能补齐。"""
    text, ctype, _ = _resolve_preview(PreviewIn(text=body.text, content_type=body.content_type))
    chunks = documents.build_chunks(text, content_type=ctype)
    if not chunks:
        raise HTTPException(status_code=400, detail="这段正文切不出块,检查是不是只有标题没有正文")
    seen = await _existing_fingerprints()
    if seen is None:
        raise HTTPException(status_code=503, detail="连不上 MySQL,录入这条路走不通(查重与落库都要它)")

    kept, skipped = [], []
    for c in chunks:
        fp = _fingerprint(c.questions, c.answer)
        if fp in seen:
            skipped.append({"questions": c.questions, "answer": c.answer[:80]})
            continue
        seen.add(fp)
        kept.append(c)

    ids = await dualwrite.write_pending(kept) if kept else []
    vectorized = None
    if body.vectorize and ids:
        try:
            vectorized = await dualwrite.vectorize_pending()
        except Exception as e:      # 向量化失败不回滚:块已在 MySQL 记 pending,补跑一次就齐
            raise HTTPException(
                status_code=502,
                detail=(f"已入库 {len(ids)} 块(pending),向量化失败:{type(e).__name__}: {e}。"
                        "修好嵌入服务后按「向量化待补块」补齐,不必重录")) from e
    return {"content_type": ctype, "chunks": len(chunks), "inserted": len(ids),
            "skipped": len(skipped), "skipped_samples": skipped[:5], "ids": ids,
            "vectorized": vectorized, "milvus": await milvus_state(),
            "chunk_stats": await repository.knowledge_stats()}


@router.post("/vectorize")
async def vectorize() -> dict:
    """把库里所有 pending 块补齐向量(幂等可重跑)。故意中断建库再按这里,漏的块会被捡起来。
    与 make kb-vectorize 调的是同一个 dualwrite.vectorize_pending,不存在两套逻辑。"""
    try:
        n = await dualwrite.vectorize_pending()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"向量化失败:{type(e).__name__}: {e}") from e
    return {"vectorized": n, "chunk_stats": await repository.knowledge_stats(),
            "milvus": await milvus_state()}


class SearchIn(BaseModel):
    q: str
    strategy: str = "vector"        # 本章只跑 dense 单路;bm25/hybrid 是 ch04 的事
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/search")
async def search(body: SearchIn) -> dict:
    """检索自测:换个说法问一句,看召回的是不是该召回的那块。
    「邮费是多少」召回运费说明,靠的是语义相近而不是词面命中——这一条在页面上现场看得见。"""
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="问一句话再检索")
    if body.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"strategy 只能是 {'/'.join(STRATEGIES)}")
    try:
        hits = await retrieval.search_knowledge(q, strategy=body.strategy, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"检索失败({type(e).__name__}: {e});嵌入上游与 Milvus 都要在") from e
    return {"q": q, "strategy": body.strategy, "top_k": body.top_k,
            "hits": [{"id": h.get("id"), "question": h.get("question"),
                      "answer": h.get("answer"), "score": h.get("score"),
                      "rerank_score": h.get("rerank_score"),
                      "section_path": h.get("section_path"),
                      "content_type": h.get("content_type"),
                      "category": h.get("category")} for h in hits]}


@router.get("/staging")
async def staging(limit: int = 30) -> dict:
    """抽 QA 暂存表:分批抽出 → 整体去重 → kept 待人审。kept 那一列就是待审队列。"""
    stats = await repository.staging_stats()
    rows = {}
    for st in ("extracted", "kept", "discarded", "approved", "rejected"):
        rows[st] = [{"id": r.id, "batch_no": r.batch_no, "source_ref": r.source_ref,
                     "question": r.question, "answer": r.answer[:160]}
                    for r in (await repository.list_staging_by_status(st))[:limit]]
    return {"stats": stats, "rows": rows}


class StagingReviewIn(BaseModel):
    ids: list[int] = Field(min_length=1, description="要处理的暂存行 id")


@router.post("/staging/approve")
async def staging_approve(body: StagingReviewIn) -> dict:
    """人工采纳:选中的暂存行写进 knowledge_chunks 并向量化,成功才置 approved。

    这是挖知识唯一的入库口。kb-mine 自己不写库——模型从聊天记录里归纳出来的问答对,
    质量参差(只对单笔订单成立、夹带订单号、把「稍等我看看」当答案),得人过一眼。

    只认 status='kept' 的行:重复点、或者拿已弃用的行来入库,都在这一步被挡掉。
    向量化失败要把刚插入的 pending 行删掉再报错,不然重试会造出重复知识(同 review.py)。"""
    rows = await repository.list_staging_by_ids(body.ids, status="kept")
    if not rows:
        raise HTTPException(status_code=409, detail="这些行不在待审状态(可能已被处理过)")

    chunks = [documents.Chunk(category="历史对话", questions=r.question, answer=r.answer,
                              section_path="mined", content_type="mined",
                              is_key_clause=documents.is_key(r.question, r.answer))
              for r in rows]
    ids: list[int] = []
    try:
        ids = await dualwrite.write_pending(chunks)
        await dualwrite.vectorize_pending()
    except Exception as e:
        logger.exception("采纳写回知识库失败 staging=%s(状态不变,可重试)", [r.id for r in rows])
        if ids:
            try:
                await repository.delete_knowledge_chunks(ids)
            except Exception:
                logger.exception("回滚 pending chunk 失败 ids=%s(需人工清理)", ids)
        raise HTTPException(
            status_code=502,
            detail=f"写回知识库失败({type(e).__name__}),这几条仍是待审,修好嵌入/Milvus 再点一次") from e

    await repository.set_staging_status([r.id for r in rows], "approved")
    logger.info("挖知识人工采纳 staging=%s → knowledge_chunks %s", [r.id for r in rows], ids)
    return {"approved": len(rows), "chunk_ids": ids}


@router.post("/staging/reject")
async def staging_reject(body: StagingReviewIn) -> dict:
    """人工弃用:不入库,只记结论。留着行不删,batch_no/source_ref 是溯源用的。"""
    rows = await repository.list_staging_by_ids(body.ids, status="kept")
    if not rows:
        raise HTTPException(status_code=409, detail="这些行不在待审状态(可能已被处理过)")
    await repository.set_staging_status([r.id for r in rows], "rejected")
    return {"rejected": len(rows)}
