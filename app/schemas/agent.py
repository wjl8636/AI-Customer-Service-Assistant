# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    user_id: str = Field(min_length=1, description="用户标识")
    message: str = Field(min_length=1, description="用户本轮消息")
    conversation_id: int | None = Field(default=None, description="续接会话,首轮留空")


class ToolCallView(BaseModel):
    id: str
    name: str
    args: dict


class ToolResultView(BaseModel):
    tool_call_id: str
    name: str
    ok: bool
    content: str


class AgentResponse(BaseModel):
    conversation_id: int
    answer: str
    tool_calls: list[ToolCallView]
    tool_results: list[ToolResultView]
    suggested_actions: list = Field(default_factory=list)
    interrupt: dict | None = Field(
        default=None,
        description="中断负载(含 type:select_order 带 orders / confirm_ticket 带 preview);无则 None")
