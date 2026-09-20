# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.extract import AfterSalesTicket, ExtractRequest, RequestType


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


def test_extract_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text="")


def test_ticket_order_id_nullable():
    t = AfterSalesTicket(order_id=None, request_type="退款", expected_solution="退全款")
    assert t.order_id is None
    assert t.request_type is RequestType.REFUND


def test_ticket_rejects_unknown_request_type():
    with pytest.raises(ValidationError):
        AfterSalesTicket(order_id=None, request_type="砍价", expected_solution="x")


def test_ticket_normalizes_placeholder_order_id_to_none():
    for placeholder in ("null", "None", "无", " N/A ", ""):
        t = AfterSalesTicket(
            order_id=placeholder, request_type="投诉", expected_solution="x"
        )
        assert t.order_id is None, placeholder
    assert AfterSalesTicket(
        order_id="MH20260701123", request_type="退款", expected_solution="x"
    ).order_id == "MH20260701123"
