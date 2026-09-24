from app.graph.build import _builder


def test_graph_has_all_nodes():
    g = _builder().compile()
    names = set(g.get_graph().nodes)
    for n in ["resolve_reference", "classify_intent", "retrieve_knowledge", "confidence_check",
              "main_agent", "agent_tools", "complaint_reply", "script_reply",
              "fallback_reply", "fetch_order", "retrieve_policy", "log"]:
        assert n in names, f"缺节点 {n}"
    assert "chitchat_reply" not in names, "chitchat_reply 应已被 script_reply 取代"


def test_graph_compiles_with_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver
    g = _builder().compile(checkpointer=InMemorySaver())
    assert g is not None
