"""ch09 审核后台 API:待审队列列表/详情/通过(写回知识库)/驳回。
通过 = 核准答案以 QA chunk 走 ch03 落库流程(write_pending → vectorize_pending 同步),
向量化成功才置「通过」——保证审核页点了通过,下一问就能检索命中(验收 3)。"""
import logging

from fastapi import APIRouter, HTTPException

from app.db import repository
from app.kb import dualwrite
from app.kb.documents import Chunk, is_key
from app.schemas.review import ApproveRequest

logger = logging.getLogger(__name__)
router = APIRouter()


async def _write_chunks(chunks: list[Chunk]) -> list[int]:
    return await dualwrite.write_pending(chunks)


async def _vectorize() -> int:
    return await dualwrite.vectorize_pending()


def _item_out(r) -> dict:
    return {"id": r.id, "normalized_question": r.normalized_question,
            "ai_suggested_answer": r.ai_suggested_answer,
            "occurrence_count": r.occurrence_count, "review_status": r.review_status,
            "created_at": r.created_at.isoformat() if r.created_at else None}


@router.get("/api/review/queue")
async def review_queue(status: str | None = None) -> dict:
    """列表按 occurrence_count 降序:一个缺口被越多用户反复问到,越该优先补(README)。"""
    rows = await repository.list_review_queue(status)
    return {"items": [_item_out(r) for r in rows]}


@router.get("/api/review/{review_id}")
async def review_detail(review_id: int) -> dict:
    """详情带归并原话 + 各自的召回快照:审核人对着快照判断「真缺知识,还是有但没检到」。"""
    detail = await repository.get_review_detail(review_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="缺口不存在")
    item, raws = detail
    out = _item_out(item)
    out["approved_answer"] = item.approved_answer
    out["raws"] = [{"raw_question": r.raw_question, "source": r.source, "reason": r.reason,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "retrieved_chunks": r.retrieved_chunks}
                   for r in raws]
    return out


@router.post("/api/review/{review_id}/approve")
async def approve(review_id: int, req: ApproveRequest) -> dict:
    detail = await repository.get_review_detail(review_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="缺口不存在")
    item, _ = detail
    if item.review_status != "待审":
        raise HTTPException(status_code=409, detail=f"当前状态为「{item.review_status}」,不可再审")

    chunk = Chunk(
        category="飞轮沉淀", questions=item.normalized_question, answer=req.approved_answer,
        section_path=f"飞轮沉淀 / {item.normalized_question}", content_type="faq",
        is_key_clause=is_key(item.normalized_question, req.approved_answer),
    )
    chunk_ids: list[int] = []
    try:
        chunk_ids = await _write_chunks([chunk])
        await _vectorize()
    except Exception:
        logger.exception("审核写回知识库失败 review=%s(状态保持待审,可重试)", review_id)
        if chunk_ids:
            # 回滚本次插入的 pending 行:vectorize_pending 处理全部 pending,
            # 残留会让重试造出重复知识条目(spec 8.2「整体回滚」)
            try:
                await repository.delete_knowledge_chunks(chunk_ids)
            except Exception:
                logger.exception("回滚 pending chunk 失败 ids=%s(需人工清理)", chunk_ids)
        raise HTTPException(status_code=502, detail="写回知识库失败(检查嵌入上游/Milvus),状态未变可重试")

    if not await repository.update_review_status(review_id, "通过", approved_answer=req.approved_answer):
        # 并发窗口:KB 已写入但状态被他人先变更——如实报出,不装成功
        logger.warning("审核状态更新落空(并发变更?)review=%s,KB 已写入 %s", review_id, chunk_ids)
        raise HTTPException(status_code=409, detail="状态已被他人变更;知识已写入,请人工核对")
    logger.info("审核通过 review=%s → knowledge_chunks %s(已向量化,下一问可检索)", review_id, chunk_ids)
    return {"ok": True, "chunk_ids": chunk_ids}


@router.post("/api/review/{review_id}/reject")
async def reject(review_id: int) -> dict:
    if not await repository.update_review_status(review_id, "驳回"):
        detail = await repository.get_review_detail(review_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="缺口不存在")
        raise HTTPException(status_code=409, detail="仅待审状态可驳回")
    return {"ok": True}
