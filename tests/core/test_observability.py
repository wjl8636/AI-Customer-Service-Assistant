"""ch09 观测挂载:缺配置不挂回调、不碰 langfuse;齐配置才 with_config。"""
from app.core import observability


def test_disabled_when_keys_missing(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    monkeypatch.setattr("app.config.settings.langfuse_secret_key", "sk")
    monkeypatch.setattr("app.config.settings.langfuse_base_url", "http://x")
    assert observability.langfuse_enabled() is False


def test_attach_returns_graph_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    sentinel = object()
    assert observability.attach_observability(sentinel) is sentinel


def test_tag_intent_noop_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    observability.tag_intent("商品咨询", 0.9)   # 不许抛异常、不许要求 langfuse 已装好服务


def test_get_langfuse_none_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    assert observability.get_langfuse() is None


def test_attach_wraps_with_callbacks_when_enabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "pk")
    monkeypatch.setattr("app.config.settings.langfuse_secret_key", "sk")
    monkeypatch.setattr("app.config.settings.langfuse_base_url", "http://localhost:3000")

    calls = {}

    class FakeGraph:
        def with_config(self, cfg):
            calls["cfg"] = cfg
            return "wrapped"

    monkeypatch.setattr(observability, "_init_client", lambda: object())
    monkeypatch.setattr(observability, "_make_handler", lambda: "HANDLER")
    assert observability.attach_observability(FakeGraph()) == "wrapped"
    assert calls["cfg"] == {"callbacks": ["HANDLER"]}
