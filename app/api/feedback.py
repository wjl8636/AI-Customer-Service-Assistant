# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 飞轮入口3:用户反馈没解决。👎 落 low_confidence_questions(source=user_feedback),
快照尽力回捞(checkpointer 里最近一轮的 retrieved_snapshot,问题对得上才用);👍 只记日志。"""
import logging

from fastapi import APIRouter

from app.db import repository
from app.graph import runtime
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _same_question(a: str, b: str) -> bool:
    return "".join(a.split()) == "".join(b.split())


@router.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    if req.rating == "up":
        logger.info("feedback up conv=%s q=%r(只记日志,不落库)", req.conversation_id, req.question[:40])
        return FeedbackResponse(pooled=False)

    snapshot = None
    try:
        turn = await runtime.get_turn_snapshot(req.conversation_id)
        if turn["snapshot"] and _same_question(turn["question"], req.question):
            snapshot = turn["snapshot"]
    except Exception:
        logger.warning("feedback 快照回捞失败 conv=%s(照常落池)", req.conversation_id, exc_info=True)

    await repository.insert_low_confidence(
        req.conversation_id, req.question, "user_feedback", "用户反馈未解决",
        retrieved_chunks=snapshot,
    )
    logger.info("feedback down conv=%s 已落池 快照=%s", req.conversation_id,
                "有" if snapshot else "无")
    return FeedbackResponse(pooled=True)
