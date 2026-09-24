"""ch07 后台分段摘要:轮结束后按层 2 用量触发,asyncio 后台跑,不阻塞当轮回复。

一次压一批,产出**新的一段**追加进 conversation_summaries,已有段落不回炉重压。
注入时带最近几段——再老的梗概对当前问题基本没用。
失败只 log 不重试:触发条件仍满足,下一轮自然重来。"""
import asyncio
import logging
import time

from pydantic import BaseModel, Field

from app.config import settings
from app.core import budget, llm, memory
from app.core.llm import get_chat_model
from app.core.prompts import SUMMARY_PROMPT
from app.db import repository

logger = logging.getLogger(__name__)

_running: dict[int, asyncio.Task] = {}   # cid -> 在跑任务(防抖:在跑不重复起)


class _Summary(BaseModel):
    summary: str = Field(description="早期对话滚动摘要,只含事实与诉求")


async def summarize_dialog(old_summary: str, dialog: str) -> str:
    """把这一批对话压成一段。严格 JSON(structured output)。

    old_summary 只作为背景给模型看,不参与重写:产出的是**新的一段**,已有段落不回炉。
    这样订单号这类事实只被压一次,不会因为反复压缩被磨掉。提示词那边也得跟着写死
    「不要复述、不要合并」,不然模型会把旧梗概重写一遍,存是分段存了,内容照样在累积损失。"""
    model = llm.structured(_Summary, slot="summary")
    r: _Summary = await (SUMMARY_PROMPT | model).ainvoke(
        {"old_summary": old_summary or "(无)", "dialog": dialog})
    return (r.summary or "").strip()


async def run_summary(conversation_id: int) -> None:
    """任务体:读消息 → 算边界 → 压出一段 → 追加。异常往外抛,由 done 回调统一 log。"""
    t0 = time.monotonic()
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        return
    msgs = await repository.list_dialog_messages(conversation_id)
    old_upto = conv.summary_upto_msg_id or 0
    # 摘要吃的就是层 2,边界就是层 1 起点,不另算一套。以前这里按轮数算边界,那是双层
    # 时代的规则:滑窗写死 3000 token 时,「留最近 K 轮、更早的压掉」和看用量是一回事。
    # 改成从窗口倒推之后两者脱节,按轮数那条每次都伸得更远,一口把还在层 1 的内容吞掉。
    boundary = conv.layer1_from_msg_id or 0
    if boundary <= old_upto:
        logger.info("summary skip conv=%s 层2 是空的(边界=%s upto=%s)",
                    conversation_id, boundary, old_upto)
        return
    seg = [m for m in msgs if old_upto < m.id <= boundary]
    dialog = "\n".join(
        f"{'用户' if m.role == 'user' else '客服'}:{m.content}" for m in seg if m.content)
    logger.info("summary start conv=%s msgs=%s upto %s->%s",
                conversation_id, len(seg), old_upto, boundary)
    summary = await summarize_dialog(conv.summary or "", dialog)
    if not summary:
        raise ValueError("摘要为空,放弃更新")
    n = await repository.append_summary_segment(
        conversation_id, from_msg_id=old_upto + 1, upto_msg_id=boundary, content=summary)
    logger.info("summary done conv=%s 第%s段 upto=%s len=%s cost=%.0fms",
                conversation_id, n, boundary, len(summary), (time.monotonic() - t0) * 1000)


async def _should_summarize(conversation_id: int, conv) -> bool:
    """要不要压一段:层 2 超预算就压,只有这一条判据。

    层 2 装的就是「降级下来、还没压」的那批,它超了预算才轮到摘要出手。以前还并着
    一条「新增满 N 条就压」的定时器,那是双层时代的规则,留到三层里会坏事:层 2 刚
    装进东西就被它清空,中间层等于白设。压缩是成本不是美德,装得下就别压。

    判据看用量不数条数。工具结果压根不落 messages 表,数条数看不见它涨,而工具越
    密集的会话恰恰涨得最快。"""
    upto = conv.summary_upto_msg_id or 0
    layer1_from = conv.layer1_from_msg_id or 0
    if layer1_from <= upto:                      # 层 2 是空的,没有可压的
        return False
    msgs = await repository.list_dialog_messages(conversation_id)
    layer2_tok = memory.chars_to_tokens(
        sum(len(m.content or "") for m in msgs if upto < m.id <= layer1_from))
    l2_budget = int(budget.compute().sliding * (1 - settings.layer1_ratio))
    if layer2_tok <= l2_budget:
        return False
    logger.info("summary trigger conv=%s 层2 约%s token > 预算 %s",
                conversation_id, layer2_tok, l2_budget)
    return True


async def maybe_schedule_summary(conversation_id: int) -> None:
    """轮结束后调用:新增消息数达阈值且无在跑任务 → create_task 后台跑,不 await(不阻塞回复)。
    本函数在用户回复路径上,任何异常只 log 不外抛(触发条件仍在,下一轮重来)。"""
    try:
        if conversation_id in _running:
            return
        conv = await repository.get_conversation(conversation_id)
        if conv is None:
            return
        if not await _should_summarize(conversation_id, conv):
            return
        # 上面两个 await 期间可能有并发触发已入表(TOCTOU),入表前再查一次防双起
        if conversation_id in _running:
            return
        task = asyncio.create_task(run_summary(conversation_id))
        _running[conversation_id] = task

        def _done(t: asyncio.Task) -> None:
            if _running.get(conversation_id) is t:   # 只清自己,防误清后来者
                _running.pop(conversation_id, None)
            if not t.cancelled() and t.exception() is not None:
                logger.error("summary failed conv=%s", conversation_id, exc_info=t.exception())

        task.add_done_callback(_done)
    except Exception:
        logger.exception("summary trigger check failed conv=%s", conversation_id)
