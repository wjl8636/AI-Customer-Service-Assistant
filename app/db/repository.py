# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from datetime import datetime

from sqlalchemy import case, func, select, update

import app.db.base as db          # 用模块属性引用,便于测试 monkeypatch async_session
from app.config import settings
from app.db.models import (
    Conversation,
    ConversationSummary,
    EvalRun,
    FaithCase,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    ReviewQueue,
    Ticket,
    ToolAuditLog,
    TopicClassification,
)

_TICKET_SEQ = 0


def _gen_ticket_no() -> str:
    global _TICKET_SEQ
    _TICKET_SEQ += 1
    return f"T{datetime.now():%Y%m%d%H%M%S}{_TICKET_SEQ:03d}"


async def create_conversation(user_id: str) -> int:
    async with db.async_session() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        await s.commit()
        return conv.id


async def get_conversation(conversation_id: int) -> Conversation | None:
    async with db.async_session() as s:
        return await s.get(Conversation, conversation_id)


async def append_message(
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with db.async_session() as s:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        s.add(msg)
        await s.commit()
        return msg.id


async def list_messages(conversation_id: int) -> list[Message]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())


async def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
    ticket_no = _gen_ticket_no()
    async with db.async_session() as s:
        s.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "已转人工"
        await s.commit()
    return ticket_no


# ---- ch07 会话上下文管理(滑窗 + 异步摘要)----


async def count_messages_after(conversation_id: int, after_id: int | None) -> int:
    """距上次摘要新增了多少条(after_id 空 = 从未摘要,数全量)。摘要触发判据。
    注:此处数全角色,摘要源 list_dialog_messages 只取 user/assistant——当前一致
    (tool 不落库);若未来 tool 行落库,两处口径需同步收紧。"""
    async with db.async_session() as s:
        q = (select(func.count()).select_from(Message)
             .where(Message.conversation_id == conversation_id))
        if after_id:
            q = q.where(Message.id > after_id)
        return int((await s.execute(q)).scalar_one())


async def list_dialog_messages(conversation_id: int) -> list[Message]:
    """user/assistant 消息按 id 升序(摘要任务源数据;tool 行不落库,过滤一道保险)。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id,
                   Message.role.in_(("user", "assistant")))
            .order_by(Message.id)
        )
        return list(result.scalars())


async def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    """某用户的会话列表(新在前,带首问预览 + 有无摘要标记),前端多会话切换用。"""
    async with db.async_session() as s:
        convs = list((await s.execute(
            select(Conversation).where(Conversation.user_id == user_id)
            .order_by(Conversation.id.desc()).limit(limit)
        )).scalars())
        out = []
        for c in convs:
            first = (await s.execute(
                select(Message.content)
                .where(Message.conversation_id == c.id, Message.role == "user")
                .order_by(Message.id).limit(1)
            )).scalar()
            out.append({
                "id": c.id, "status": c.status,
                "preview": (first or "(空会话)")[:40],
                "has_summary": bool(c.summary),
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            })
        return out


async def update_conversation_summary(conversation_id: int, summary: str, upto_msg_id: int) -> None:
    """摘要成功后原子更新两字段(一起写,不会出现摘要新边界旧)。"""
    async with db.async_session() as s:
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.summary = summary
            conv.summary_upto_msg_id = upto_msg_id
            await s.commit()


async def update_layer1_from(conversation_id: int, msg_id: int) -> None:
    """层 1 起点往后挪。只在超预算时挪一次,两次之间不动——边界稳,渲染才逐字节稳。"""
    async with db.async_session() as s:
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.layer1_from_msg_id = msg_id
            await s.commit()


async def append_summary_segment(conversation_id: int, from_msg_id: int,
                                 upto_msg_id: int, content: str) -> int:
    """追加一段摘要,同时把 conversations 的边界和拼好的投影一起更新。

    段落只追加不改写:已有的段不会再进模型,里面的订单号这类事实压过一次就定死了。
    conversations.summary 存的是最近若干段拼好的结果,读的时候少一次查询。"""
    async with db.async_session() as s:
        # 先把已有的几段读出来再 add:反过来的话 autoflush 会把新段也算进这次查询,
        # 拼投影时它就出现两次。
        prev = (await s.execute(
            select(ConversationSummary.seq, ConversationSummary.content)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.seq.desc())
            .limit(max(settings.summary_inject_segments - 1, 0))
        )).all()
        seq = (prev[0].seq if prev else 0) + 1
        s.add(ConversationSummary(conversation_id=conversation_id, seq=seq,
                                  from_msg_id=from_msg_id, upto_msg_id=upto_msg_id,
                                  content=content))
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.summary = "\n".join([*(r.content for r in reversed(prev)), content])
            conv.summary_upto_msg_id = upto_msg_id
        await s.commit()
        return seq


async def list_summary_segments(conversation_id: int, limit: int | None = None) -> list[str]:
    """取最近几段,按时间正序返回。再老的梗概对当前问题基本没用,不必全带。"""
    n = settings.summary_inject_segments if limit is None else limit
    async with db.async_session() as s:
        rows = (await s.execute(
            select(ConversationSummary.content)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.seq.desc())
            .limit(n)
        )).scalars().all()
        return list(reversed(rows))


# ---- ch03 知识库 chunk / 暂存表 ----


async def insert_knowledge_chunk(
    category: str, questions: str, answer: str,
    section_path: str | None = None, content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session() as s:
        row = KnowledgeChunk(
            category=category, questions=questions, answer=answer,
            section_path=section_path, content_type=content_type,
            is_key_clause=is_key_clause,
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"
            await s.commit()


async def set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id
            await s.commit()


async def list_chunks_by_content_types(content_types) -> list[KnowledgeChunk]:
    """按 content_type 取块(文档类建库材料)。飞轮写回的块 content_type 不在这几类里,
    所以按这个条件取到的就是「来自 data/kb/*.md 的那部分」,补库改文件时只动它们。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.content_type.in_(list(content_types)))
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def repend_chunk_text(chunk_id: int, questions: str, answer: str) -> None:
    """改写一块的正文并退回 pending,等 vectorize_pending 重嵌。

    为什么不是「删掉重插」:Milvus 那边 upsert 的主键就是这个 id,原地重嵌等于原地更新,
    引用角标、prev/next 邻居指针、飞轮写回的块都不受影响。整库重建(kb-reset)会连飞轮
    审核通过写回的那些块一起清掉,补库这种小改动犯不上。"""
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.questions = questions
            row.answer = answer
            row.vectorize_status = "pending"
            await s.commit()


async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())


async def list_all_questions() -> list[str]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions))
        return list(result.scalars())


async def count_chunks_by_content_types(content_types) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.content_type.in_(list(content_types)))
        )
        return int(result.scalar_one())


async def knowledge_stats() -> dict:
    """知识库盘点:总块数 + 双写状态分布 + 内容类型分布 + 关键条款数。
    录入页的库存概览一次拿齐,不必让页面发四五个请求各问一个数。"""
    async with db.async_session() as s:
        total = (await s.execute(select(func.count()).select_from(KnowledgeChunk))).scalar_one()
        by_status = {k: int(v) for k, v in (await s.execute(
            select(KnowledgeChunk.vectorize_status, func.count())
            .group_by(KnowledgeChunk.vectorize_status))).all()}
        by_type = {(k or "未标注"): int(v) for k, v in (await s.execute(
            select(KnowledgeChunk.content_type, func.count())
            .group_by(KnowledgeChunk.content_type))).all()}
        key_clause = (await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.is_key_clause == 1))).scalar_one()
    return {"total": int(total), "pending": by_status.get("pending", 0),
            "done": by_status.get("done", 0), "by_content_type": by_type,
            "key_clause": int(key_clause)}


async def list_recent_chunks(limit: int = 20) -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit)
        )
        return list(result.scalars())


async def list_chunk_pairs() -> list[tuple[str, str]]:
    """全表的 (questions, answer) 对,给录入查重算指纹。
    只按问法查重会误杀同一节切出来的多块——表格按行拆、超长散文递归切,这些块共用节标题,
    问法一模一样、正文各不相同,是该留的。"""
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions, KnowledgeChunk.answer))
        return [(q, a) for q, a in result.all()]


async def staging_stats() -> dict:
    """抽 QA 暂存表盘点:三个状态各几条 + 批次数 + 最近批次号。
    「先进暂存、整体去重再入库」这一步在页面上要看得见,靠的就是这三个状态的落差。"""
    async with db.async_session() as s:
        counts = {k: int(v) for k, v in (await s.execute(
            select(QaExtractionStaging.status, func.count())
            .group_by(QaExtractionStaging.status))).all()}
        batches = (await s.execute(
            select(func.count(func.distinct(QaExtractionStaging.batch_no))))).scalar_one()
        latest = (await s.execute(
            select(QaExtractionStaging.batch_no)
            .order_by(QaExtractionStaging.id.desc()).limit(1))).scalar_one_or_none()
    return {"counts": {k: counts.get(k, 0)
                       for k in ("extracted", "kept", "discarded")},
            "total": sum(counts.values()), "batches": int(batches), "latest_batch": latest}


async def insert_staging(batch_no: str, source_ref: str | None, question: str, answer: str) -> int:
    async with db.async_session() as s:
        row = QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())


async def list_staging_by_ids(ids: list[int], status: str | None = None) -> list[QaExtractionStaging]:
    """按 id 取暂存行,可再限定状态。人工采纳/弃用时用它——限定 status='kept' 就能挡住
    重复采纳(点两次只有第一次拿得到行)和拿已弃用的行来入库。"""
    if not ids:
        return []
    async with db.async_session() as s:
        q = select(QaExtractionStaging).where(QaExtractionStaging.id.in_(ids))
        if status is not None:
            q = q.where(QaExtractionStaging.status == status)
        return list((await s.execute(q.order_by(QaExtractionStaging.id))).scalars())


async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(QaExtractionStaging, i)
            if row is not None:
                row.status = status
        await s.commit()


async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None,
    retrieved_chunks: list | None = None,
) -> int:
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason, retrieved_chunks=retrieved_chunks,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    """挖知识用:每个会话及其按序消息(供拼成对话文本喂 LLM)。"""
    async with db.async_session() as s:
        conv_ids = list(
            (await s.execute(select(Conversation.id).order_by(Conversation.id))).scalars()
        )
        out: list[tuple[int, list[Message]]] = []
        for cid in conv_ids:
            msgs = list(
                (await s.execute(
                    select(Message).where(Message.conversation_id == cid).order_by(Message.id)
                )).scalars()
            )
            out.append((cid, msgs))
        return out


# ---- ch09 数据飞轮(待审队列 + 评估轮次)----


async def fetch_unmatched_low_conf(limit: int) -> list[LowConfidenceQuestion]:
    """飞轮处理游标:尚未归并(matched_review_id IS NULL)的池内问题,按 id 升序。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(LowConfidenceQuestion.id).limit(limit)
        )
        return list(result.scalars())


async def list_review_candidates(limit: int = 200) -> list[dict]:
    """查重候选:全部状态的缺口行(用户拍板:比全部,驳回即终审),updated_at 倒序截断防 token 爆。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(ReviewQueue.id, ReviewQueue.normalized_question)
            .order_by(ReviewQueue.updated_at.desc()).limit(limit)
        )
        return [{"id": r.id, "normalized_question": r.normalized_question} for r in result]


async def insert_review_item(normalized_question: str, ai_suggested_answer: str | None) -> int:
    async with db.async_session() as s:
        row = ReviewQueue(normalized_question=normalized_question,
                          ai_suggested_answer=ai_suggested_answer)
        s.add(row)
        await s.commit()
        return row.id


async def increment_occurrence(review_id: int) -> None:
    """查重命中:只累加次数,状态不动(命中已驳回/已通过也一样——驳回即终审)。
    原子 UPDATE,不走读改写,两批飞轮重叠时不丢计数。"""
    async with db.async_session() as s:
        await s.execute(
            update(ReviewQueue)
            .where(ReviewQueue.id == review_id)
            .values(occurrence_count=ReviewQueue.occurrence_count + 1)
        )
        await s.commit()


async def set_matched_review(lcq_id: int, review_id: int) -> None:
    async with db.async_session() as s:
        row = await s.get(LowConfidenceQuestion, lcq_id)
        if row is not None:
            row.matched_review_id = review_id
            await s.commit()


async def list_review_queue(status: str | None) -> list[ReviewQueue]:
    """审核页列表:按出现次数降序(频次=优先级),同频新的在前。"""
    async with db.async_session() as s:
        q = select(ReviewQueue).order_by(
            ReviewQueue.occurrence_count.desc(), ReviewQueue.id.desc())
        if status:
            q = q.where(ReviewQueue.review_status == status)
        return list((await s.execute(q)).scalars())


async def get_review_detail(review_id: int) -> tuple[ReviewQueue, list[LowConfidenceQuestion]] | None:
    """详情:缺口行 + 归并进来的原话流水(带 source/快照,审核人判断真缺还是没检到)。"""
    async with db.async_session() as s:
        item = await s.get(ReviewQueue, review_id)
        if item is None:
            return None
        raws = list((await s.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id == review_id)
            .order_by(LowConfidenceQuestion.id)
        )).scalars())
        return item, raws


async def update_review_status(review_id: int, status: str, approved_answer: str | None = None) -> bool:
    """仅「待审」可流转(通过/驳回都是终态);返回是否真的更新了。"""
    async with db.async_session() as s:
        row = await s.get(ReviewQueue, review_id)
        if row is None or row.review_status != "待审":
            return False
        row.review_status = status
        if approved_answer is not None:
            row.approved_answer = approved_answer
        await s.commit()
        return True


async def delete_knowledge_chunks(ids: list[int]) -> None:
    """审核写回失败的回滚:删掉本次插入的 pending chunk,防重试造出重复知识条目。"""
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(KnowledgeChunk, i)
            if row is not None:
                await s.delete(row)
        await s.commit()


async def insert_eval_run(triggered_by: str, dataset_size: int, metrics: dict) -> int:
    async with db.async_session() as s:
        row = EvalRun(triggered_by=triggered_by, dataset_size=dataset_size, metrics=metrics)
        s.add(row)
        await s.commit()
        return row.id


async def list_eval_runs(limit: int = 10) -> list[EvalRun]:
    async with db.async_session() as s:
        result = await s.execute(select(EvalRun).order_by(EvalRun.id.desc()).limit(limit))
        return list(result.scalars())


# ---- ch08 工具审计 ----


async def insert_tool_audit(
    conversation_id: int | None,
    tool_call_id: str | None,
    tool_name: str,
    tool_source: str,
    mcp_server: str | None,
    arguments: dict | None,
    result_summary: str | None,
    status: str,
    error_message: str | None,
    retry_count: int,
    duration_ms: int | None,
) -> None:
    """工具调用审计落一条。调用方(engine)自行 try/except——审计失败不许反拦工具执行。"""
    async with db.async_session() as s:
        s.add(ToolAuditLog(
            conversation_id=conversation_id, tool_call_id=tool_call_id,
            tool_name=tool_name, tool_source=tool_source, mcp_server=mcp_server,
            arguments=arguments, result_summary=result_summary, status=status,
            error_message=error_message, retry_count=retry_count, duration_ms=duration_ms,
        ))
        await s.commit()


# ---------- ch10 主题分类 ----------

def _pool_text_stmt():
    """池问题取文本的公共查询:优先归并后的标准化问法,没归并退回原话。"""
    return (
        select(LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question,
               ReviewQueue.normalized_question)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(LowConfidenceQuestion.id)
    )


async def list_pool_texts() -> list[dict]:
    """ch10 训练语料捞取:全量低置信度问题。"""
    async with db.async_session() as s:
        rows = (await s.execute(_pool_text_stmt())).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm in rows]


async def list_unclassified_questions(limit: int = 500) -> list[dict]:
    """ch10 旁路批处理捞取:尚未归类、且已归并的低置信度问题。
    分类器只吃归并阶段产出的标准化问法;未归并(无 matched_review_id)的
    不进分类,留到下一轮归并后再归——保证输入永远是语义完整的单句,不喂原话。"""
    stmt = (
        _pool_text_stmt()
        .outerjoin(TopicClassification,
                   TopicClassification.question_id == LowConfidenceQuestion.id)
        .where(TopicClassification.id.is_(None))
        .where(LowConfidenceQuestion.matched_review_id.is_not(None))
        .limit(limit)
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm in rows]


async def insert_topic_classifications(rows: list[dict]) -> int:
    """批量写归类结果;rows: [{question_id, labels}]。"""
    async with db.async_session() as s:
        s.add_all([TopicClassification(question_id=r["question_id"], labels=r["labels"])
                   for r in rows])
        await s.commit()
    return len(rows)


async def topic_distribution(samples_per_class: int = 3) -> dict:
    """主题分布:17 类各自问题量 + 每类样例;labels JSON 在 Python 侧聚合(量级小)。"""
    from app.core.taxonomy import TOPIC_NAMES

    stmt = (
        select(TopicClassification.labels, LowConfidenceQuestion.raw_question,
               ReviewQueue.normalized_question, TopicClassification.classified_at)
        .join(LowConfidenceQuestion,
              TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()
    counts = {name: 0 for name in TOPIC_NAMES}
    samples: dict[str, list[str]] = {name: [] for name in TOPIC_NAMES}
    latest = None
    for labels, raw, norm, ts in rows:
        text = norm or raw
        latest = ts if latest is None or ts > latest else latest
        for lb in labels or []:
            if lb in counts:
                counts[lb] += 1
                # 样例去重:归并后同一句标准化问法会对应池里好几行,三条样例都一样就白给了
                if len(samples[lb]) < samples_per_class and text not in samples[lb]:
                    samples[lb].append(text)
    return {
        "total": len(rows),
        "latest": latest.isoformat() if latest else None,
        "classes": [{"label": n, "count": counts[n], "samples": samples[n]}
                    for n in TOPIC_NAMES],
    }


async def topic_questions(label: str, page: int = 1, size: int = 20) -> dict:
    """某一类目下已归类的问题,分页。总数与分布页那张图同源,都是数 labels 命中。

    labels 是 JSON 数组,筛选与分页都在 Python 侧做:归类总量是问题池规模(百级),
    一次全捞不贵,换来的是不依赖 MySQL 的 JSON 函数,sqlite 上跑测试也一致。
    """
    stmt = (
        select(TopicClassification.question_id, TopicClassification.labels,
               TopicClassification.classified_at, LowConfidenceQuestion.raw_question,
               LowConfidenceQuestion.source, LowConfidenceQuestion.created_at,
               ReviewQueue.normalized_question, ReviewQueue.occurrence_count,
               ReviewQueue.review_status)
        .join(LowConfidenceQuestion,
              TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(TopicClassification.question_id.desc())
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()

    hit = [r for r in rows if label in (r[1] or [])]
    size = max(1, min(size, 100))
    pages = max(1, -(-len(hit) // size))
    page = max(1, min(page, pages))
    items = [
        {"question_id": qid, "labels": labels,
         "text": norm or raw, "raw_question": raw,
         "normalized": norm is not None,
         "source": source, "occurrence_count": occ,
         "review_status": status,
         "asked_at": asked.isoformat() if asked else None,
         "classified_at": ts.isoformat() if ts else None}
        for qid, labels, ts, raw, source, asked, norm, occ, status
        in hit[(page - 1) * size: page * size]
    ]
    return {"label": label, "total": len(hit), "page": page, "size": size,
            "pages": pages, "items": items}


# ---- ch04 编造个案台账 ----


async def upsert_faith_case(eval_id: str, bucket: str, query: str, answer: str, reason: str,
                            citations: list | None = None, strategy: str = "hybrid_rerank",
                            judge_model: str | None = None) -> tuple[int, bool]:
    """一题一行:已有就更新成最近一次并 seen_count+1,返回 (id, 是否复发)。

    复发 = 这条之前被标过「已解决 / 无需解决」,现在又被判编造。那不是新问题,而是上次
    没改对(或者改动被回退),所以状态退回「未解决」重新进待处理列表——resolved_at 留着,
    列表上据此打「复发」标。
    """
    async with db.async_session() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id))).scalar_one_or_none()
        now = datetime.now()
        if row is None:
            row = FaithCase(eval_id=eval_id, bucket=bucket, query=query, answer=answer,
                            reason=reason, citations=citations, strategy=strategy,
                            judge_model=judge_model, status="未解决", seen_count=1,
                            first_seen_at=now, last_seen_at=now)
            s.add(row)
            await s.commit()
            return row.id, False
        reopened = row.status != "未解决"
        row.bucket, row.query, row.answer, row.reason = bucket, query, answer, reason
        row.strategy, row.judge_model = strategy, judge_model
        if citations is not None:
            row.citations = citations
        row.seen_count = (row.seen_count or 0) + 1
        row.last_seen_at = now
        row.status = "未解决"
        await s.commit()
        return row.id, reopened


async def list_faith_cases(status: str | None = None, page: int = 1,
                           size: int = 5) -> tuple[list[FaithCase], int, dict]:
    """分页列表 + 三个状态各自的条数。未解决排前面,同状态内最近判出的排前面。"""
    async with db.async_session() as s:
        counts = {k: 0 for k in ("未解决", "已解决", "无需解决")}
        for st, n in (await s.execute(
                select(FaithCase.status, func.count()).group_by(FaithCase.status))).all():
            counts[st] = int(n)
        q = select(FaithCase)
        if status:
            q = q.where(FaithCase.status == status)
        total = int((await s.execute(
            select(func.count()).select_from(q.subquery()))).scalar_one())
        rows = (await s.execute(
            q.order_by(case((FaithCase.status == "未解决", 0), else_=1),
                       FaithCase.last_seen_at.desc(), FaithCase.id.desc())
            .limit(size).offset(max(0, (page - 1) * size)))).scalars()
        return list(rows), total, counts


async def faith_case_status_map() -> dict[str, str]:
    """eval_id → 处置状态。给幻觉率算「本轮」用:报告只给这一轮判出了哪几题,
    每题现在是什么处置得回台账查——台账是跨轮累计的,不能直接拿它的总数当本轮判出。"""
    async with db.async_session() as s:
        rows = (await s.execute(select(FaithCase.eval_id, FaithCase.status))).all()
        return {eid: st for eid, st in rows}


async def set_faith_case_status(case_id: int, status: str,
                                resolution: str | None = None) -> FaithCase | None:
    """人工处置:已解决 / 无需解决 / 退回未解决。

    处置要留交代:标「已解决」得写清怎么解决的,标「无需解决」得写清为什么不用改——
    半年后回头看,一个没写理由的「无需解决」和没处理过没区别。校验在 API 层(能回 400),
    这里只负责落库:退回未解决时连 resolved_at 和说明一起清掉。
    """
    async with db.async_session() as s:
        row = await s.get(FaithCase, case_id)
        if row is None:
            return None
        row.status = status
        if status == "未解决":
            row.resolved_at, row.resolution = None, None
        else:
            row.resolved_at = datetime.now()
            row.resolution = resolution
        await s.commit()
        await s.refresh(row)
        return row
