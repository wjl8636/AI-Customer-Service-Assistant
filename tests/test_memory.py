from langchain_core.messages import AIMessage, HumanMessage

from app.core.memory import SessionStore, trim_history


def test_store_get_unknown_session_returns_empty():
    assert SessionStore().get("nope") == []


def test_store_append_and_get_isolated_by_session():
    store = SessionStore()
    store.append("a", HumanMessage("hi"), AIMessage("hello"))
    store.append("b", HumanMessage("嗨"))
    assert len(store.get("a")) == 2
    assert len(store.get("b")) == 1


def test_trim_keeps_recent_and_starts_on_human():
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(f"问题{i}:" + "喵" * 50))
        msgs.append(AIMessage(f"回答{i}:" + "喵" * 50))
    trimmed = trim_history(msgs, max_tokens=200)
    assert 0 < len(trimmed) < len(msgs)
    assert trimmed[0].type == "human"
    assert trimmed[-1] == msgs[-1]


def test_trim_noop_when_under_budget():
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    assert trim_history(msgs, max_tokens=2000) == msgs


# ---- ch07 锚点切窗 + 摘要注入 ----

from langchain_core.messages import SystemMessage

from app.core.memory import build_window, summary_line, summary_system


def _dialog(n_turns: int, start_db_id: int = 1) -> list:
    """n 轮对话,用户消息带 db-id 锚点(每轮占 2 个 id:user、assistant)。"""
    msgs = []
    db_id = start_db_id
    for i in range(n_turns):
        msgs.append(HumanMessage(f"问题{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(f"回答{i}"))
        db_id += 2
    return msgs


def test_build_window_cuts_at_anchor():
    msgs = _dialog(10)                       # 用户消息 db-1,3,...,19
    win = build_window(msgs, summary_upto_msg_id=8, max_tokens=100000)
    assert win[0].id == "db-9"               # 第一条 > 8 的用户消息
    assert len(win) == 12                    # 第 5-10 轮共 6 轮
    assert win[-1] == msgs[-1]


def test_build_window_no_boundary_keeps_all():
    msgs = _dialog(3)
    assert build_window(msgs, summary_upto_msg_id=0, max_tokens=100000) == msgs


def test_build_window_anchor_missing_degrades_to_trim():
    msgs = [HumanMessage("旧消息无锚点"), AIMessage("答")] * 3   # ch07 前的旧会话消息
    win = build_window(list(msgs), summary_upto_msg_id=99, max_tokens=100000)
    assert len(win) == 6                     # 找不到锚点 → 不切,整段进 trim


def test_build_window_token_cap_still_applies():
    """token 上限仍然是最后一道兜底。预算现在按模型窗口算(240k 级),
    要验证裁剪逻辑本身就得显式传一个小上限,不然这批消息根本装得下。"""
    msgs = []
    for k in range(20):
        msgs.append(HumanMessage(f"问题{k}" + "喵" * 300, id=f"db-{k * 2 + 1}"))
        msgs.append(AIMessage("答" * 300))
    win = build_window(msgs, 0, 0, max_tokens=800)
    assert 0 < len(win) < len(msgs)

def test_build_window_never_returns_empty():
    msgs = [HumanMessage("超长" + "喵" * 5000, id="db-1")]
    win = build_window(msgs, summary_upto_msg_id=0, max_tokens=10)
    assert win == msgs                       # 裁到空则宁可超预算也回退原窗


def test_summary_helpers():
    assert summary_line(None) == "" and summary_line("") == ""
    assert "订单1001" in summary_line("用户问过订单1001")
    assert summary_system(None) is None
    ss = summary_system("用户问过订单1001")
    assert isinstance(ss, SystemMessage) and "订单1001" in ss.content
