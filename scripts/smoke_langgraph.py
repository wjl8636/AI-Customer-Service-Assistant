"""ch05 红线冒烟:StateGraph + AsyncSqliteSaver + 多模式流式 是否在当前版本跑通,并打印流式形状。"""
import asyncio
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.llm import get_chat_model


class S(TypedDict):
    messages: Annotated[list, add_messages]


async def call_model(state: S):
    ai = await get_chat_model(streaming=True).ainvoke(state["messages"])
    return {"messages": [ai]}


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        b = StateGraph(S)
        b.add_node("call_model", call_model)
        b.add_edge(START, "call_model")
        b.add_edge("call_model", END)
        graph = b.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": "smoke-1"}}
        seen_modes = set()
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage("用一句话介绍你自己")]},
            config, stream_mode=["messages", "updates"],
        ):
            seen_modes.add(mode)
            if mode == "messages":
                msg, meta = chunk
                print("MSG node=", meta.get("langgraph_node"), "text=", (msg.content or "")[:20])
            else:
                print("UPD", list(chunk.keys()))
        print("OK modes=", seen_modes)


if __name__ == "__main__":
    asyncio.run(main())
