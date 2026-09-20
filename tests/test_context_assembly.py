# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch07 上下文拼装(纯函数级):拼装顺序固定、滑窗从摘要边界后接原文、无摘要不插额外 system。"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.nodes import _agent_messages, _history_text


def _state(n_turns=3, summary="", upto=0):
    msgs = []
    db_id = 1
    for i in range(n_turns):
        msgs.append(HumanMessage(f"问题{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(f"回答{i}"))
        db_id += 2
    return {"messages": msgs, "summary": summary, "summary_upto_msg_id": upto}


def test_agent_messages_order_with_summary():
    """拼装顺序固定:人设 system 打头 → 滑窗原文 → 摘要与本轮材料紧跟最后一条用户消息。

    摘要不再单独占一条 system:上游模板会把所有 system 上提合并,摘要每会话不同,
    放 system 里会把工具 schema 挤到可变内容之后,前缀缓存整段作废(实测 2048→0)。
    """
    ms = _agent_messages(_state(summary="用户问过订单1001", upto=0))
    assert isinstance(ms[0], SystemMessage) and "小喵" in ms[0].content   # 人设红线打头
    assert sum(isinstance(m, SystemMessage) for m in ms) == 1            # 全列表只此一条
    assert isinstance(ms[1], HumanMessage)
    ctx = [m for m in ms if "订单1001" in (m.content or "")]
    assert ctx and not isinstance(ctx[0], SystemMessage)                 # 摘要走用户侧


def test_agent_messages_no_summary_no_extra_system():
    ms = _agent_messages(_state(summary="", upto=0))
    assert isinstance(ms[0], SystemMessage)
    assert not isinstance(ms[1], SystemMessage)     # 无摘要不插第二条 system


def test_agent_messages_window_cut_by_boundary():
    ms = _agent_messages(_state(n_turns=6, summary="早期摘要", upto=6))
    win = [m for m in ms if not isinstance(m, SystemMessage)]
    assert win[0].id == "db-7"                      # 滑窗从边界后第一条用户消息接原文


def test_history_text_prepends_summary_and_windows():
    text = _history_text(_state(n_turns=6, summary="用户问过订单1001", upto=6))
    assert text.startswith("(早前对话摘要:用户问过订单1001")
    assert "问题0" not in text                       # 边界前原文不出现
    assert "问题3" in text                           # 窗内原文在(排除当前最后一条 human)


def test_history_text_no_summary_same_as_before():
    text = _history_text(_state(n_turns=2))
    assert "摘要" not in text and "问题0" in text


def test_log_model_context_shows_summary_and_window(caplog):
    import logging
    from app.graph.nodes import _agent_messages, _log_model_context
    state = _state(n_turns=6, summary="用户问过订单1001,留了手机13800138000", upto=6)
    msgs = _agent_messages(state)
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        _log_model_context(state, msgs)
    text = caplog.text
    assert "model_ctx" in text
    assert "订单1001" in text                 # 摘要全文可见
    assert "[human] '问题3" in text           # 滑窗每条消息角色+预览可见
    assert "问题0" not in text                # 边界前原文不进上下文,也不该出现在日志


def _layered_state(layer1_from: int):
    """三轮对话,客服答复都远超层 2 的截断阈值(60 字),便于看出压没压。"""
    long_reply = "这是一段很长的客服答复" * 12          # 132 字
    msgs = []
    for i, db_id in enumerate((1, 3, 5)):
        msgs.append(HumanMessage(f"问题{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(long_reply + f"第{i}轮"))
    return {"messages": msgs, "summary": "", "summary_upto_msg_id": 0,
            "layer1_from_msg_id": layer1_from}


def test_层1锚点必须真的传进渲染_否则三层跑成两层():
    """`build_window` 支持三层,但节点不传第三个参数的话它会静默退化成两层。

    这条钉的是「接线」不是「函数」:build_window 自己的三层行为在 test_layers.py 里
    测过了,漏的是 _agent_messages 有没有把 State 里的 layer1_from_msg_id 递给它。
    参数有默认值 0,不传不报错、日志照打、锚点照落库,只是层 2 那一刀永远不切——
    跑起来还是两层,而且从外面一点看不出来。
    """
    ms = _agent_messages(_layered_state(layer1_from=3))
    replies = [m.content for m in ms if isinstance(m, AIMessage)]
    assert len(replies) == 3
    assert replies[0].endswith("…(略)")      # 前两轮在 db-3 之前/之上,属层 2,被截短
    assert replies[1].endswith("…(略)")
    assert not replies[2].endswith("…(略)")  # 最后一轮属层 1,一字不压


def test_不给层1锚点时维持全原文():
    """锚点为 0 就是「层 2 为空」,所有消息都算层 1,一条都不压。"""
    ms = _agent_messages(_layered_state(layer1_from=0))
    replies = [m.content for m in ms if isinstance(m, AIMessage)]
    assert not any(r.endswith("…(略)") for r in replies)


def test_coref历史与主力Agent用同一套锚点():
    """两处渲染口径必须一致,否则指代消解看到的历史和模型看到的不是同一份。"""
    text = _history_text(_layered_state(layer1_from=3))
    assert "…(略)" in text
