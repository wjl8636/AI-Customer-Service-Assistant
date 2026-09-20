# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from typing import Literal

from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: Literal["up", "down"]
    question: str          # 被评价那轮的用户原话(前端从气泡上带上来)


class FeedbackResponse(BaseModel):
    ok: bool = True
    pooled: bool           # down 且成功落池才为 True
