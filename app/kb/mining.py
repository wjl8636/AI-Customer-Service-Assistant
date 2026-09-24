import logging
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db import repository
from app.kb import dedup


class QaPair(BaseModel):
    question: str
    answer: str


class QaExtraction(BaseModel):
    """LLM 结构化输出 schema:用两个并列标量数组一一对应。
    实测 glm-5.2 的 anthropic 兼容上游对嵌套对象数组(list[对象])的工具 schema 会 502,
    标量 list[str] 正常,故不用 list[QaPair],改并列数组规避。"""

    questions: list[str] = Field(
        default_factory=list, description="问法列表,与 answers 一一对应;无可复用问答则为空"
    )
    answers: list[str] = Field(
        default_factory=list, description="答案列表,与 questions 一一对应"
    )


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    model = model or get_chat_model()
    chain = MINING_PROMPT | model.with_structured_output(QaExtraction)
    result: QaExtraction = await chain.ainvoke(
        {"conversations": "\n---\n".join(conversation_texts)}
    )
    qs, ans = result.questions, result.answers
    if len(qs) != len(ans):
        # 并列数组本应一一对应;长度不一致(模型偶发)按较短对齐,记警告不静默错位
        logger.warning("挖知识并列数组长度不一致 questions=%d answers=%d,按较短对齐", len(qs), len(ans))
    n = min(len(qs), len(ans))
    return [QaPair(question=qs[i], answer=ans[i]) for i in range(n)]


async def _load_conversation_texts() -> list[tuple[str, str]]:
    """返回 [(source_ref, 对话文本)];对话文本由该会话的 user/assistant 消息拼成。"""
    convs = await repository.list_conversations_with_messages()
    out: list[tuple[str, str]] = []
    for conv_id, msgs in convs:
        lines = [f"{m.role}: {m.content}" for m in msgs if m.content]
        if lines:
            out.append((f"conv:{conv_id}", "\n".join(lines)))
    return out


async def mine(batch_size: int = 20, model=None) -> dict:
    """可重跑 job:对话 → 分批 LLM 抽取写 staging → 整体去重 → kept 停在暂存表等人审。

    **不直接写 knowledge_chunks**:抽出来的问答对是模型从聊天记录里归纳的,客服当时那句
    回答可能只对那一单成立,可能带着具体订单号和收货地址,也可能只是一句「稍等我帮您看看」。
    这种东西进了库,之后每次检索都可能被捞出来当依据。所以挖完就停,人在 /kb 页面
    「对话挖知识」那张卡片上逐条采纳或弃用,采纳的才入库(见 app/api/kb.py 的 staging 两个端点)。
    """
    sources = await _load_conversation_texts()
    batch_no = datetime.now().strftime("mine-%Y%m%d%H%M%S")
    # 1) 分批抽取 → 写 staging(extracted)
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        for p in pairs:
            await repository.insert_staging(batch_no, batch[0][0], p.question, p.answer)
    # 2) 整体去重(仅本批 extracted 内 + 对已有 knowledge;圈定 batch_no 避免卷入历史崩溃残留)
    staged = [s for s in await repository.list_staging_by_status("extracted") if s.batch_no == batch_no]
    existing = await repository.list_all_questions()
    kept, discarded = dedup.dedupe(staged, existing)
    await repository.set_staging_status([s.id for s in kept], "kept")
    await repository.set_staging_status([s.id for s in discarded], "discarded")
    # 3) kept 就是待审队列,入库交给人工。这里不再落 knowledge_chunks
    return {"sources": len(sources), "extracted": len(staged),
            "kept": len(kept), "discarded": len(discarded)}
