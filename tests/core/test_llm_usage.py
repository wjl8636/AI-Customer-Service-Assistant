"""ch09 Cost Control 前提:所有模型调用(含流式)都回传 usage,按意图统计才不漏账。"""
from app.core.llm import get_chat_model


def test_streaming_model_requests_usage():
    m = get_chat_model(streaming=True)
    assert m.stream_usage is True


def test_non_streaming_model_also_flags_usage():
    assert get_chat_model().stream_usage is True
