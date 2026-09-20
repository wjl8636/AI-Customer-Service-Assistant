# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from app.config import settings
from app.core.llm import needs_non_streaming_tools, resolve_slot


def test_chat_slot_reads_chat_group():
    model, base, key = resolve_slot("chat")
    assert (model, base, key) == (
        settings.chat_model, settings.chat_base_url, settings.chat_api_key)


def test_empty_slot_falls_back_to_chat(monkeypatch):
    """三项都留空时，意图槽位完全继承 CHAT——只用一家上游就一行都不用填。"""
    monkeypatch.setattr(settings, "intent_model", "")
    monkeypatch.setattr(settings, "intent_base_url", "")
    monkeypatch.setattr(settings, "intent_api_key", "")
    assert resolve_slot("intent") == (
        settings.chat_model, settings.chat_base_url, settings.chat_api_key)


def test_slot_overrides_are_independent(monkeypatch):
    """只换模型名不换地址，或只换地址不换模型名，都该按项生效。"""
    monkeypatch.setattr(settings, "intent_model", "small-model")
    monkeypatch.setattr(settings, "intent_base_url", "")
    monkeypatch.setattr(settings, "intent_api_key", "")
    assert resolve_slot("intent") == (
        "small-model", settings.chat_base_url, settings.chat_api_key)

    monkeypatch.setattr(settings, "summary_base_url", "https://other.example/v1")
    monkeypatch.setattr(settings, "summary_api_key", "sk-other")
    monkeypatch.setattr(settings, "summary_model", "")
    assert resolve_slot("summary") == (
        settings.chat_model, "https://other.example/v1", "sk-other")


def test_needs_non_streaming_tools_reads_slot(monkeypatch):
    """槽位换了模型，这个判断要跟着走 —— 之前这里漏传 slot，跑起来直接 NameError。"""
    monkeypatch.setattr(settings, "intent_model", "MiniMax-M3")
    assert needs_non_streaming_tools(slot="intent")
    monkeypatch.setattr(settings, "intent_model", "deepseek-v4-flash")
    assert not needs_non_streaming_tools(slot="intent")


def test_needs_non_streaming_tools_explicit_model_wins():
    assert needs_non_streaming_tools("MiniMax-M2.7")
    assert not needs_non_streaming_tools("gpt-4o")
