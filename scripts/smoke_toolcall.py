# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""真实验证 glm-5.2 直连上游能否返回结构化 tool_calls。go/no-go 风险闸。"""
import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model


class AddInput(BaseModel):
    a: int = Field(description="第一个加数")
    b: int = Field(description="第二个加数")


@tool(args_schema=AddInput)
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b


async def main() -> None:
    model = get_chat_model()  # 非流式,直连 settings.chat_base_url
    bound = model.bind_tools([add])
    ai = await bound.ainvoke("请用工具计算 23 加 19 等于多少")
    print("content:", repr(ai.content))
    print("tool_calls:", ai.tool_calls)
    if ai.tool_calls and ai.tool_calls[0]["name"] == "add":
        print(f"✅ GO:{settings.chat_model} 支持 tool calling,选中 add,args=", ai.tool_calls[0]["args"])
    else:
        print("❌ NO-GO:未返回预期 tool_calls —— 停下来问用户,不自行换方案")


if __name__ == "__main__":
    asyncio.run(main())
