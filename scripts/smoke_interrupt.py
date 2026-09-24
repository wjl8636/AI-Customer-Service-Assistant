"""ch06 红线冒烟:interrupt/Command(resume) 在当前 langgraph 上的中断 surface 形状。
不调聊天上游——只用一个纯 interrupt 节点。用法:.venv/bin/python -m scripts.smoke_interrupt

钉死三件事(供 Task 7 fetch_order / Task 13 runtime 落地):
  A) ainvoke 遇中断返回态 __interrupt__ 键结构与取值路径
  B) Command(resume=v) 续跑,v 成为 interrupt() 返回值
  C) astream(stream_mode=["messages","updates"]) 中断出现在哪种 chunk(vs C' aget_state 探 pending)
  D) 直接调用被中断节点(无 resume 上下文)抛的 GraphInterrupt 载荷取法
"""
import asyncio
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


class S(TypedDict):
    messages: Annotated[list, add_messages]
    picked: str


async def ask_order(state: S):
    picked = interrupt({"type": "select_order",
                        "orders": [{"order_id": "1001"}, {"order_id": "2002"}]})
    return {"picked": picked}


async def confirm(state: S):
    return {"messages": [("ai", f"已选订单 {state['picked']}")]}


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        b = StateGraph(S)
        b.add_node("ask_order", ask_order)
        b.add_node("confirm", confirm)
        b.add_edge(START, "ask_order")
        b.add_edge("ask_order", "confirm")
        b.add_edge("confirm", END)
        graph = b.compile(checkpointer=cp)

        # A) 非流式:首跑应带 __interrupt__
        config = {"configurable": {"thread_id": "smoke-int-1"}}
        out = await graph.ainvoke({"messages": [("human", "我要退款")]}, config)
        print("A ainvoke keys=", list(out.keys()))
        print("A __interrupt__=", out.get("__interrupt__"))
        if out.get("__interrupt__"):
            print("A first.value=", out["__interrupt__"][0].value)

        # B) 非流式 resume
        out2 = await graph.ainvoke(Command(resume="1001"), config)
        print("B resume picked=", out2.get("picked"), "msgs=", [m.content for m in out2["messages"]])

        # C) 流式:中断信息出现在哪种 chunk
        config2 = {"configurable": {"thread_id": "smoke-int-2"}}
        print("C astream chunks:")
        async for mode, chunk in graph.astream(
                {"messages": [("human", "我要退款")]}, config2,
                stream_mode=["messages", "updates"]):
            if mode == "updates":
                print("  UPD keys=", list(chunk.keys()), "val=", chunk)
        # C') 流式后探 pending(备用探测路径)
        snap = await graph.aget_state(config2)
        print("C' aget_state .next=", snap.next,
              "interrupts=", getattr(snap, "interrupts", None),
              "tasks_interrupts=", [t.interrupts for t in snap.tasks])
        # C'') 流式 resume
        print("C'' astream resume:")
        async for mode, chunk in graph.astream(
                Command(resume="2002"), config2, stream_mode=["messages", "updates"]):
            print("  ", mode, chunk if mode == "updates"
                  else (chunk[0].content, chunk[1].get("langgraph_node")))

        # D) 直接调用被中断节点(无 runnable 上下文)——记录 interrupt() 在图外的真实行为
        from langgraph.errors import GraphInterrupt
        try:
            await ask_order({"messages": []})
            print("D no error(unexpected)")
        except GraphInterrupt as e:
            print("D GraphInterrupt first.value=", e.args[0][0].value if e.args else None)
        except RuntimeError as e:
            print("D RuntimeError(图外不能调 interrupt):", e)
            print("D 结论:fetch_order 的缺单 interrupt 路径须经【编译图 ainvoke】测,"
                  "不能直接调节点函数(直接调会 RuntimeError,非 GraphInterrupt)。")


if __name__ == "__main__":
    asyncio.run(main())
