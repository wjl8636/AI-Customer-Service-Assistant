import pytest
from langchain_core.messages import AIMessage

from app.config import settings
from app.graph.routing import (
    confidence_gate, route_by_intent, should_continue, INTENT_TO_ROUTE,
)


@pytest.mark.parametrize("intent,expect", [
    ("投诉", "escalate"),
    ("闲聊", "fallback_script"),
    ("其他", "fallback_script"),
    ("商品咨询", "knowledge"),
    ("退款退货", "refund_flow"),
    ("售后", "refund_flow"),
    ("人工", "business"),
    ("物流", "business"),
    ("订单", "business"),
])
def test_route_by_intent_five_outlets(intent, expect):
    assert route_by_intent({"intent": intent}) == expect


def test_route_by_intent_unknown_defaults_business():
    assert route_by_intent({"intent": "火星语"}) == "business"
    assert route_by_intent({}) == "business"


def test_intent_to_route_covers_nine_classes():
    assert set(INTENT_TO_ROUTE) == {
        "投诉", "闲聊", "其他", "商品咨询", "退款退货", "售后", "人工", "物流", "订单"}


def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"
    assert confidence_gate({}) == "weak"


def test_should_continue_stops_when_no_tool_calls():
    state = {"messages": [AIMessage("答完了")], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_should_continue_continues_on_tool_calls():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "continue"


def test_should_continue_stops_on_step_cap():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": settings.max_agent_steps, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_token_花销不再当停止条件():
    # 环里只按步数封顶。token 记账继续走(进 trace、给 ch09 统计),但不参与判断:
    # 混在这里卡,阈值估偏一次就把整条工具链掐断,而且掐在模型已经决定调工具之后
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 10_000_000}
    assert should_continue(state) == "continue"
