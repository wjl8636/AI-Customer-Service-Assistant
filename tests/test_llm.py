from app.config import settings
from app.core.llm import _thinking_kwargs, get_chat_model


def test_factory_points_to_chat_upstream():
    m = get_chat_model()
    assert m.model_name == settings.chat_model  # 跟随配置,不硬编码具体模型名
    # 直连聊天上游,不经网关。三组上游各配一份 base_url,这里必须拿到 chat 那一组
    assert str(m.openai_api_base) == settings.chat_base_url


def test_factory_streaming_flag():
    assert get_chat_model(streaming=True).streaming is True


def test_思考链三项留空就一个都不传(monkeypatch):
    # 留空 = 用上游默认,不能替调用方塞参数
    for k in ("chat_thinking", "chat_reasoning_effort", "chat_reasoning_split"):
        monkeypatch.setattr(settings, k, "")
    assert _thinking_kwargs() == {}


def test_关思考链走_extra_body(monkeypatch):
    # thinking 不是 OpenAI 协议标准字段,必须走 extra_body 才到得了上游
    monkeypatch.setattr(settings, "chat_thinking", "disabled")
    monkeypatch.setattr(settings, "chat_reasoning_effort", "")
    monkeypatch.setattr(settings, "chat_reasoning_split", "")
    assert _thinking_kwargs() == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_effort_是标准字段_不进_extra_body(monkeypatch):
    monkeypatch.setattr(settings, "chat_thinking", "adaptive")
    monkeypatch.setattr(settings, "chat_reasoning_effort", "high")
    monkeypatch.setattr(settings, "chat_reasoning_split", "true")
    kw = _thinking_kwargs()
    assert kw["reasoning_effort"] == "high"                    # 顶层
    assert kw["extra_body"]["thinking"] == {"type": "adaptive"}  # extra_body
    assert kw["extra_body"]["reasoning_split"] is True          # 字符串转成布尔


def test_reasoning_split_认常见的几种真值写法(monkeypatch):
    # 只认字面量 "true" 会坑人:填 1 / yes / on 的人本意是开,却被转成 False 显式发出去,
    # 等于把「思考链拆到独立字段」按死。而这项正是关不掉思考链时的兜底,发反了
    # <think> 就留在 content 里,结构化解析当场被带崩
    monkeypatch.setattr(settings, "chat_thinking", "")
    monkeypatch.setattr(settings, "chat_reasoning_effort", "")
    for v in ("true", "True", "1", "yes", "on", " TRUE "):
        monkeypatch.setattr(settings, "chat_reasoning_split", v)
        assert _thinking_kwargs()["extra_body"]["reasoning_split"] is True, v
    for v in ("false", "0", "no", "off"):
        monkeypatch.setattr(settings, "chat_reasoning_split", v)
        assert _thinking_kwargs()["extra_body"]["reasoning_split"] is False, v


def test_配置真的落到模型上(monkeypatch):
    # 光拼对不够,得确认 ChatOpenAI 收下了
    monkeypatch.setattr(settings, "chat_thinking", "disabled")
    monkeypatch.setattr(settings, "chat_reasoning_effort", "")
    monkeypatch.setattr(settings, "chat_reasoning_split", "")
    m = get_chat_model()
    assert m.extra_body == {"thinking": {"type": "disabled"}}
