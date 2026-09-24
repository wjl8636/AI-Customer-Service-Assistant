import logging
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import settings
from app.core import summarizer
from app.core.observability import attach_observability
from app.config import settings
from app.core import budget, memory
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点(其余节点的模型调用 token 不进 delta)
ANSWER_NODES = {"main_agent"}
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply"}
# 产出编号引用的检索节点(知识路 retrieve_knowledge + 退款政策路 retrieve_policy)
CITATION_NODES = {"retrieve_knowledge", "retrieve_policy"}

_graph = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC


class ConversationNotFound(Exception):
    pass


async def init_graph() -> None:
    """lifespan 起:开持久 checkpointer + setup + 编译图。"""
    global _graph, _cm
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    checkpointer = await _cm.__aenter__()
    await checkpointer.setup()
    # ch09:编译时挂一次 Langfuse 回调,全图自动 trace(未配置时原样返回,零依赖)
    _graph = attach_observability(build_graph(checkpointer=checkpointer))
    logger.info("图已编译,checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    global _graph, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None


def get_graph():
    if _graph is None:
        raise RuntimeError("图未初始化,应在 FastAPI lifespan 调 init_graph()")
    return _graph


async def get_turn_snapshot(conversation_id: int) -> dict:
    """👎 快照尽力回捞(ch09):读该会话 checkpointer 终态,返回最近一轮的用户问题与召回快照。
    调用方拿去比对被踩的 question,对得上才用快照——踩历史消息时快照可能已是别轮的,不硬塞。"""
    config = {"configurable": {"thread_id": str(conversation_id)}}
    st = await get_graph().aget_state(config)
    values = getattr(st, "values", None) or {}
    question = ""
    for m in reversed(values.get("messages", [])):
        if getattr(m, "type", "") == "human":
            question = m.content if isinstance(m.content, str) else ""
            break
    return {"question": question, "snapshot": values.get("retrieved_snapshot") or []}


def dedup_actions(actions: list) -> list:
    """跨多个 ReAct 轮次累积的 suggested_actions 可能重复(如 agent 多次提议
    create_ticket);按 type 去重,保留每种 type 首次出现的那条。"""
    seen = set()
    out = []
    for action in actions:
        t = action.get("type")
        if t in seen:
            continue
        seen.add(t)
        out.append(action)
    return out


async def _ensure_conversation(user_id: str,
                               conversation_id: int | None) -> tuple[int, str, int, int]:
    """返回 (cid, summary, summary_upto_msg_id, layer1_from_msg_id):
    入口一次库读把摘要和两个分层锚点一并带出。"""
    if conversation_id is None:
        return await repository.create_conversation(user_id), "", 0, 0
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        raise ConversationNotFound(conversation_id)
    return (conversation_id, conv.summary or "",
            conv.summary_upto_msg_id or 0, conv.layer1_from_msg_id or 0)


def _graph_input(user_id: str, message: str, cid: int, msg_id: int,
                 summary: str, summary_upto: int, layer1_from: int = 0) -> dict:
    # 显式重置每轮的输出通道:checkpointer 按 thread_id 持久化 State,这些无 reducer 的标量
    # 会跨轮残留(如上一轮闲聊/投诉的 answer、建工单可选项串到本轮),须在入口清零。
    # messages 有 add_messages reducer(追加=跨轮历史),不清;steps/tokens_used 是本轮预算。
    # ch07:用户消息带 db-{msg_id} 锚点(滑窗与摘要边界对齐);summary 两字段每轮从库刷新。
    return {"messages": [HumanMessage(message, id=f"db-{msg_id}")], "user_id": user_id,
            "conversation_id": cid, "steps": 0, "tokens_used": 0,
            "answer": "", "suggested_actions": [], "evidence": "",
            "evidence_strong": False, "citations": [],
            "evidence_confidence": 0.0, "fallback_source": "", "retrieved_snapshot": [],
            "resolved_query": "", "intent_confidence": 0.0,
            "order_id": "", "order_data": {},
            "summary": summary, "summary_upto_msg_id": summary_upto,
            "layer1_from_msg_id": layer1_from,
            "trace": None}  # merge 通道:None=重置哨兵(见 merge_dict),清上一轮 trace 残留


def _interrupt_payload(state: dict):
    """从终态提取中断负载(整个 __interrupt__[0].value:含 type + orders/preview 等);
    无中断返回 None。ch08 起中断不止订单选择器,负载透传、消费方按 type 分发。"""
    intr = state.get("__interrupt__")
    if not intr:
        return None
    return intr[0].value


async def _final_state(cid: int) -> dict:
    """流式跑完后从 checkpointer 取终态。astream 不返回终态,分层判断要靠它。"""
    snap = await get_graph().aget_state({"configurable": {"thread_id": str(cid)}})
    return snap.values if snap else {}


async def settle_layers(cid: int, state: dict | None,
                        summary_upto: int, layer1_from: int) -> None:
    """轮末结算分层:层 1 超预算就把起点往后挪一批。

    纯计算,不调模型,不阻塞。挪一次挪到位、两次之间不动——边界稳住,渲染才逐字节
    稳定,前缀缓存才保得住;每轮微调等于每轮重写中间内容,缓存天天作废。

    state 传 None 时自己去 checkpointer 取(流式路径拿不到终态)。整段吞异常:
    这是轮末的锦上添花,不能影响已经发出去的回复。"""
    try:
        if state is None:
            state = await _final_state(cid)
        msgs = (state or {}).get("messages") or []
        if not msgs:
            return
        l1_budget = int(budget.compute().sliding * settings.layer1_ratio)
        _, layer1_tok = memory.layer_tokens(msgs, summary_upto, layer1_from)
        if layer1_tok <= l1_budget:
            return
        new_from = memory.next_layer1_from(msgs, summary_upto, l1_budget)
        if new_from > layer1_from:
            await repository.update_layer1_from(cid, new_from)
            logger.info("层1 降级 conv=%s %s→%s(层1 %s > 预算 %s)",
                        cid, layer1_from, new_from, layer1_tok, l1_budget)
    except Exception:
        logger.exception("分层结算失败 conv=%s(不影响本轮回复)", cid)


async def run_turn(user_id, message, conversation_id) -> dict:
    """非流式:落 user 消息 → ainvoke → 返回终态(供 /api/agent、eval)。遇 interrupt 带 orders。"""
    cid, summary, upto, layer1 = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    # ch09:langfuse_session_id 让同一会话多轮在 Langfuse 界面串成一组(未挂回调时无害)
    config = {"configurable": {"thread_id": str(cid)},
              "metadata": {"langfuse_session_id": str(cid)}}
    final = await get_graph().ainvoke(
        _graph_input(user_id, message, cid, msg_id, summary, upto, layer1), config)
    await settle_layers(cid, final, upto, layer1)  # 层1 超预算就降级一批(纯计算)
    await summarizer.maybe_schedule_summary(cid)   # ch07 轮后触发检查(后台跑,不阻塞返回)
    return {"conversation_id": cid, "state": final, "interrupt": _interrupt_payload(final)}


async def resume_turn(conversation_id: int, resume_value) -> dict:
    """非流式续跑(供 /api/agent、eval):Command(resume) 回填后跑到下一个中断或结束。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    config = {"configurable": {"thread_id": str(conversation_id)},
              "metadata": {"langfuse_session_id": str(conversation_id)}}
    final = await get_graph().ainvoke(Command(resume=resume_value), config)
    await summarizer.maybe_schedule_summary(conversation_id)
    return {"conversation_id": conversation_id, "state": final,
            "interrupt": _interrupt_payload(final)}


async def _stream_events(cid: int, stream_source) -> AsyncIterator[dict]:
    """astream 多模式 → 事件 dict。stream_source 为 _graph_input(...)(新一轮)或 Command(resume=...)(续跑)。
    遇 interrupt(updates 块含 __interrupt__,Task 1 冒烟钉死)→ 吐 interrupt 事件并结束本次流
    (前端点选后走 /api/actions/resume 续流)。"""
    config = {"configurable": {"thread_id": str(cid)},
              "metadata": {"langfuse_session_id": str(cid)}}
    actions: list = []
    async for mode, chunk in get_graph().astream(
        stream_source, config, stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    yield {"type": "delta", "text": text}
        elif mode == "updates":
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                # 带上 conversation_id:中断时不发 done 帧,前端要靠这个才知道续跑用哪个会话。
                # ch08:负载按中断类型透传(select_order 带 orders,confirm_ticket 带 preview),
                # 前端按 kind 分发。
                ev = {"type": "interrupt", "kind": payload.get("type", ""), "conversation_id": cid}
                for k in ("orders", "preview"):
                    if payload.get(k) is not None:
                        ev[k] = payload[k]
                yield ev
                return
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                if node in CITATION_NODES and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        # submit_refund 仍是「拦成前端表单」的合成消息,不发工具帧;
                        # create_ticket ch08 起确认后真执行,照常发帧
                        if name and name != "submit_refund":
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions.extend(upd["suggested_actions"])
    if actions:
        yield {"type": "actions", "items": dedup_actions(actions)}
    yield {"type": "done", "conversation_id": cid}


async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:落 user 消息 → astream 多模式 → 事件 dict(供 /api/chat 转 SSE)。"""
    cid, summary, upto, layer1 = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    async for ev in _stream_events(
            cid, _graph_input(user_id, message, cid, msg_id, summary, upto, layer1)):
        yield ev
    await settle_layers(cid, None, upto, layer1)   # 流式拿不到终态,内部自取
    await summarizer.maybe_schedule_summary(cid)   # ch07 流吐完再触发(后台跑,不阻塞)


async def stream_resume(conversation_id: int, resume_value) -> AsyncIterator[dict]:
    """前端点选订单后续跑同一会话的暂停流(Command(resume) 续 astream,事件与 stream_turn 同构)。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    async for ev in _stream_events(conversation_id, Command(resume=resume_value)):
        yield ev
    await summarizer.maybe_schedule_summary(conversation_id)
