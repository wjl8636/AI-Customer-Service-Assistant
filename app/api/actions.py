import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.db import repository
from app.graph import runtime
from app.schemas.actions import (
    CreateRefundRequest,
    CreateRefundResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    ResumeRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(req: CreateTicketRequest) -> CreateTicketResponse:
    """建工单按钮:用户点了才写 tickets 表(复用 ch02 工单能力)。转人工是另一件事,纯前端。"""
    try:
        ticket_no = await repository.create_ticket(
            req.conversation_id, req.description, req.ticket_type)
    except SQLAlchemyError:
        logger.exception("建工单失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="工单系统暂时不可用,请稍后重试")
    return CreateTicketResponse(ticket_no=ticket_no)


@router.post("/api/actions/create-refund", response_model=CreateRefundResponse)
async def create_refund_action(req: CreateRefundRequest) -> CreateRefundResponse:
    """退款表单提交:写 tickets(ticket_type='退款'),描述带订单号 + 固定类目原因。复用 ch02 工单能力。"""
    desc = f"退款申请 订单号={req.order_id} 原因={req.reason}"
    try:
        ticket_no = await repository.create_ticket(req.conversation_id, desc, "退款")
    except SQLAlchemyError:
        logger.exception("退款单创建失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="退款系统暂时不可用,请稍后重试")
    return CreateRefundResponse(ticket_no=ticket_no)


@router.post("/api/actions/resume")
async def resume_action(req: ResumeRequest):
    """中断续跑(SSE,事件与 /api/chat 同构):订单选择器点选回 order_id;
    工单预览卡回 confirmed(ch08,resume 值为 {"confirmed": bool},agent_tools 按此放行/拒绝)。"""
    if req.order_id is None and req.confirmed is None:
        raise HTTPException(status_code=400, detail="order_id 与 confirmed 至少传一个")
    resume_value = req.order_id if req.order_id is not None else {"confirmed": bool(req.confirmed)}

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in runtime.stream_resume(req.conversation_id, resume_value):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "interrupt":
                    yield _sse({"event": "interrupt",
                                **{k: v for k, v in ev.items() if k != "type"}})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except runtime.ConversationNotFound:
            yield "event: error\n"
            yield _sse({"message": "会话不存在"})
            return
        except Exception:
            logger.exception("续跑失败 conv=%s", req.conversation_id)
            yield "event: error\n"
            yield _sse({"message": "上游暂时不可用,请稍后重试"})
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
