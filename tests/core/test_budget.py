from app.config import settings
from app.core import budget


def test_large_window_limited_by_turns():
    b = budget.compute(window=1_000_000)
    assert b.limited_by == "turns"
    assert b.sliding == settings.context_budget_turns * b.per_turn
    assert b.turns == settings.context_budget_turns
    assert b.healthy


def test_small_window_shrinks_coverage():
    # 32K 现在连一轮峰值都装不下(见下面那条),换个还撑得住的窗口验「窗口项收紧」
    b = budget.compute(window=131_072)
    assert b.limited_by == "window"
    assert b.sliding == 131_072 - b.fixed - b.peak
    assert b.turns < settings.context_budget_turns


def test_window_too_small_is_unhealthy():
    b = budget.compute(window=8_192)
    assert not b.healthy
    assert b.turns == 0


def test_fixed_cost_alone_exceeding_window_yields_zero():
    b = budget.compute(window=1_000)
    assert b.sliding == 0
    assert not b.healthy


def test_fixed_cost_covers_evidence_and_output():
    b = budget.compute(window=1_000_000)
    assert b.fixed == (
        settings.system_prompt_tokens
        + budget.evidence_tokens()
        + budget.summary_tokens()
        + settings.max_output_tokens
        + settings.budget_safety_margin
    )


def test_evidence_scales_with_rerank_top_k():
    assert budget.evidence_tokens(top_k=5) * 2 == budget.evidence_tokens(top_k=10)


def test_per_turn_是历史稳态_不是瞬时峰值():
    """峰值和稳态是两个数,合成一个必有一边错。

    峰值管「这一轮跑得起来吗」:6 步的 AI 消息和工具结果同时挂在消息列表里。
    稳态管「多少轮不被裁掉」:旧轮的工具结果早被 compress_tool_result 压成一行了。
    拿峰值去乘轮数,等于给一个系统自己承诺不会出现的状态留位置。
    """
    b = budget.compute(window=1_000_000)
    assert b.per_turn == (settings.max_user_input_tokens + settings.max_output_tokens
                          + settings.max_agent_steps * settings.layer2_tool_keep_tokens)
    assert b.peak == (settings.max_user_input_tokens + settings.max_agent_steps
                      * (settings.tool_result_max_tokens + settings.agent_step_ai_tokens))
    assert b.peak > b.per_turn          # 峰值必然比稳态大,不然拆它没意义


def test_装不下一轮峰值就该报不健康():
    """这条钉的是拆分前的假阳性:32K 窗口按旧口径(只比 X+Y)报健康,可它第 4 步就撞墙。"""
    b = budget.compute(window=32_768)
    assert not b.healthy
    assert b.window - b.fixed > settings.max_user_input_tokens + settings.max_output_tokens
    assert b.window - b.fixed < b.peak      # 旧口径装得下,算上峰值就装不下


def test_lookup_window_longest_prefix_wins():
    assert budget.lookup_window("MiniMax-M3") == (1_000_000, True)
    assert budget.lookup_window("deepseek-v4-flash-0423") == (1_048_576, True)
    assert budget.lookup_window("deepseek-v4") == (1_048_576, True)


def test_lookup_window_unknown_model_falls_back():
    win, found = budget.lookup_window("some-new-model-2030")
    assert not found
    assert win == budget._FALLBACK_WINDOW


def test_lookup_window_handles_empty_name():
    win, found = budget.lookup_window("")
    assert not found and win > 0


def test_lookup_window_strips_provider_prefix():
    """硅基流动这类上游的模型名带 provider 前缀，不能直接前缀匹配。"""
    assert budget.lookup_window("deepseek-ai/DeepSeek-V4-Flash") == (1_048_576, True)
    assert budget.lookup_window("Qwen/Qwen3-32B") == (131_072, True)


def test_summary_reserve_scales_with_segments():
    assert budget.summary_tokens() == (
        settings.summary_inject_segments * settings.summary_tokens_per_segment)
