# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import logging

from app.tools import registry


def test_builtin_scan_registers_core_tools_without_query_logistics():
    names = {s.name for s in registry.builtin_specs()}
    # 钉红线不钉全集:核心五工具必须在、query_logistics 必须不在(物流由 MCP 接管)。
    # builtin/ 是"丢文件即注册"的即插即用目录(如验收 1 的 promotions.py),
    # 断言精确集合会让任何合法丢入文件打红测试,与本章语义自相矛盾。
    assert {"query_order", "query_product", "query_faq", "create_ticket", "submit_refund"} <= names
    assert "query_logistics" not in names


def test_every_spec_has_three_essentials():
    for s in registry.builtin_specs():
        assert s.name and s.description                      # 名字 + 用途描述
        assert isinstance(s.json_schema, dict) and s.json_schema.get("properties") is not None


def test_permissions_only_trust_our_side():
    by = {s.name: s for s in registry.builtin_specs()}
    assert by["create_ticket"].permission == "write"
    assert all(s.permission == "read" for n, s in by.items() if n != "create_ticket")
    assert registry.permission_for("mcp_random_tool") == "read"   # 未登记 MCP 工具默认只读放行


def test_create_ticket_schema_excludes_injected_conversation_id():
    spec = registry.get_builtin_spec("create_ticket")
    assert spec.inject_conversation is True
    assert "conversation_id" not in spec.json_schema.get("properties", {})  # 模型不见注入参数


def test_query_faq_keeps_extended_timeout():
    assert registry.get_builtin_spec("query_faq").timeout == 30.0  # RAG 管线远慢于默认 5s


def test_duplicate_register_is_dropped_with_warning(caplog):
    spec = registry.builtin_specs()[0]
    dup = registry.ToolSpec(name=spec.name, description="dup",
                            json_schema={"type": "object", "properties": {}},
                            tool=spec.tool, permission="read", source="builtin")
    with caplog.at_level(logging.WARNING):
        registry.register(dup)
    assert registry.get_builtin_spec(spec.name).description != "dup"   # 先到者保留
    assert any("重名" in r.message for r in caplog.records)
