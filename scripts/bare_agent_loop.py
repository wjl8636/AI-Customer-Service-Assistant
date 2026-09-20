# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""祛魅:不用任何框架,手写最裸的 Agent 循环——看清它就是个带工具清单的 for 循环。
用法:.venv/bin/python -m scripts.bare_agent_loop "订单1001的物流到哪了"
"""
import asyncio
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_chat_model
from app.core.prompts import AGENT_SYSTEM
from app.tools import engine, registry


async def run_agent(query: str, max_turns: int = 6):
    specs = {s.name: s for s in await registry.get_all_specs()}   # ch08:内置+MCP 现拉
    model = get_chat_model().bind_tools([s.tool for s in specs.values()])
    messages = [SystemMessage(AGENT_SYSTEM), HumanMessage(query)]
    for step in range(1, max_turns + 1):
        ai = await model.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            print(f"[第{step}步] 无工具调用 → 收敛")
            return ai.content
        for tc in ai.tool_calls:
            print(f"[第{step}步] 调用工具 {tc['name']} args={tc['args']}")
            run = await engine.execute_tool_call(tc, 0, specs)
            messages.append(run.tool_message)
    print(f"[封顶] max_turns={max_turns} 用尽仍未收敛,生产环境此处应走兜底话术")
    return "(未收敛)"


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "订单1001的物流到哪了"
    print("问题:", query)
    print("答复:", await run_agent(query))


if __name__ == "__main__":
    asyncio.run(main())
