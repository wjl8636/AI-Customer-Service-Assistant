from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.core import memory


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, description="用户标识,用于建/归属会话")
    message: str = Field(min_length=1, description="用户本轮消息")
    conversation_id: int | None = Field(
        default=None, description="续接会话时带上;为空则新建会话,由 done 帧回传新 id"
    )

    @field_validator("message")
    @classmethod
    def _within_input_budget(cls, v: str) -> str:
        """单轮输入不超过 X。上下文预算按每轮最坏 X+Y 留,放进来一条超长的就把
        这笔账冲掉了;更实际的是超长输入本身会把窗口顶爆,上游直接报错。"""
        limit = settings.max_user_input_tokens
        if memory.count_tokens([("human", v)]) > limit:
            raise ValueError(
                f"这条消息太长了(上限约 {memory.tokens_to_chars(limit)} 字),"
                f"麻烦分几次说,或者只留关键信息")
        return v
