# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""RAG 评估页的请求体:编造个案的处置状态与处置说明。"""
from typing import Literal

from pydantic import BaseModel, Field


class FaithCaseStatusRequest(BaseModel):
    status: Literal["未解决", "已解决", "无需解决"] = Field(description="处置状态")
    # 处置说明:已解决写怎么解决的,无需解决写为什么不用改;退回未解决不需要(会被清空)
    resolution: str | None = Field(default=None, max_length=300, description="处置说明")
