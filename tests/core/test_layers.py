# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import settings
from app.core import memory


def _dialogue(turns: int, reply_chars: int = 200) -> list:
    """造 turns 轮对话，用户消息带 db- 锚点。"""
    msgs = []
    for i in range(1, turns + 1):
        msgs.append(HumanMessage(f"第{i}轮的问题", id=f"db-{i * 2 - 1}"))
        msgs.append(AIMessage("答" * reply_chars))
    return msgs


def test_user_text_never_compressed():
    """用户原话一字不动——指代线索全在这半边。"""
    msgs = [HumanMessage("我之前问的那个跑鞋，订单 202603150088")]
    assert memory.to_layer2(msgs)[0].content == msgs[0].content


def test_reply_truncated_beyond_threshold():
    short, long = "好的", "答" * 500
    assert memory.compress_reply(short) == short
    out = memory.compress_reply(long)
    assert len(out) < len(long)
    assert out.startswith("答" * settings.layer2_reply_keep_chars)


def test_small_tool_result_kept_large_one_dropped():
    small = '{"order_id": "1001", "status": "已付款"}'
    assert memory.compress_tool_result("query_order", small) == small
    big = memory.compress_tool_result("query_faq", "证据" * 2000)
    assert "query_faq" in big and len(big) < 60


def test_layer2_is_idempotent():
    """降级按批发生，重算多少次结果都一样，否则前缀天天变、cache 保不住。"""
    msgs = _dialogue(3) + [ToolMessage("证据" * 2000, tool_call_id="t1", name="query_faq")]
    once = memory.to_layer2(msgs)
    twice = memory.to_layer2(once)
    assert [m.content for m in once] == [m.content for m in twice]


def test_layer2_keeps_message_ids():
    """id 是切窗锚点，压缩不能把它弄丢。"""
    msgs = _dialogue(2)
    for src, out in zip(msgs, memory.to_layer2(msgs)):
        assert out.id == src.id


def test_build_window_without_anchor_is_single_layer():
    msgs = _dialogue(3)
    assert memory.build_window(msgs, 0, 0, max_tokens=10_000) == msgs


def test_build_window_splits_by_anchor():
    """锚点之前渲染成半压，之后保持原文。"""
    msgs = _dialogue(6)
    out = memory.build_window(msgs, 0, layer1_from_msg_id=5, max_tokens=100_000)
    assert "略" in out[1].content          # 第 1 轮的答复被截断
    assert "略" not in out[-1].content     # 最近一轮的答复完整


def test_layer_tokens_splits_at_anchor():
    msgs = _dialogue(6)
    l2, l1 = memory.layer_tokens(msgs, 0, layer1_from_msg_id=5)
    assert l2 > 0 and l1 > 0
    whole = memory.count_tokens(msgs)
    assert l2 + l1 < whole                # 层 2 压过，合计小于原始体积


def test_next_layer1_from_moves_boundary_back():
    """预算越小，边界越靠后（留的原文越少）。"""
    msgs = _dialogue(10)
    tight = memory.next_layer1_from(msgs, 0, layer1_budget=500)
    loose = memory.next_layer1_from(msgs, 0, layer1_budget=5_000)
    assert tight > loose


def test_next_layer1_from_returns_zero_when_budget_is_ample():
    msgs = _dialogue(3)
    assert memory.next_layer1_from(msgs, 0, layer1_budget=10**6) == 0


def test_层2压缩必须保住tool_calls():
    """压的是正文,不是这一轮的调用结构。

    AIMessage 带着 tool_calls,后面那条 ToolMessage 拿 tool_call_id 指着它。渲染层 2 时
    新建 AIMessage 漏掉 tool_calls,这对引用就断了,上游直接回 400
    「tool result's tool id not found」,整轮回复挂掉。
    单测里造消息很容易只写 content,这条洞因此能一直藏着,直到层 2 真被接进请求路径。
    """
    from langchain_core.messages import AIMessage, ToolMessage
    call = {"name": "query_order", "args": {"order_id": "1001"}, "id": "call_abc", "type": "tool_call"}
    msgs = [AIMessage("我来帮您查一下订单" * 20, id="a1", tool_calls=[call]),
            ToolMessage("订单数据" * 200, tool_call_id="call_abc", name="query_order", id="t1")]
    out = memory.to_layer2(msgs)
    assert out[0].content.endswith("…(略)")            # 正文压了
    assert [c["id"] for c in out[0].tool_calls] == ["call_abc"]   # 调用结构没丢
    assert out[1].tool_call_id == "call_abc"
