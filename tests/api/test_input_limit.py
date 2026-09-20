# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest
from pydantic import ValidationError

from app.config import settings
from app.schemas.chat import ChatRequest


def test_normal_message_passes():
    req = ChatRequest(user_id="u1", message="那双跑鞋能退吗")
    assert req.conversation_id is None


def test_oversized_message_rejected():
    """超过 X 的输入在入口就挡下，不进上下文。"""
    huge = "订单详情" * 2000
    with pytest.raises(ValidationError) as e:
        ChatRequest(user_id="u1", message=huge)
    assert "太长" in str(e.value)


def test_limit_follows_config(monkeypatch):
    monkeypatch.setattr(settings, "max_user_input_tokens", 10)
    with pytest.raises(ValidationError):
        ChatRequest(user_id="u1", message="这句话按十个 token 的上限来说已经超了")
