# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""上下文预算:窗口按块分配,滑窗拿剩下的。

一次请求里系统提示与工具 schema、检索证据、注入的摘要段、输出预留各占一块,
滑窗只能用余下的部分。
这里有两个容易混成一个的数,合起来必错:

  峰值    turn_peak_tokens()    当前这一轮跑起来的瞬时占用。ReAct 最多走 max_agent_steps 步,
                                每步的 AI 消息和工具结果都同时挂在消息列表里喂给下一步
  稳态    history_per_turn()    这一轮沉淀进历史后的占用。层 1 边界挪过去以后,
                                compress_tool_result 会把大块工具结果压成一行

启动自检问的是「这一轮跑得起来吗」,要用峰值;滑窗覆盖几轮问的是「多少轮不被裁掉」,
要用稳态。拿峰值去乘轮数,等于给一个系统自己承诺不会出现的状态留位置。

滑窗预算取两个数的小者:

  轮数项  N * 稳态         想留住的轮数倒推
  窗口项  W - 固定开销 - 峰值   窗口扣掉固定块和本轮峰值之后,实际匀得出来的

窗口大时轮数项生效,窗口小时窗口项收紧,连一轮都装不下就判为不健康。
窗口大小按 chat_model 查表得到,配置里填了非 0 值则以配置为准。
"""
from dataclasses import dataclass

from app.config import settings

# OpenAI 兼容的 /v1/models 只返回 id/object/created/owned_by,拿不到窗口大小,
# 只能本地查表。键是模型名小写后的前缀,取最长匹配。
# 表会过期:配置里 model_context_window 填了非 0 值就以配置为准,不查表。
_KNOWN_WINDOWS: dict[str, int] = {
    "minimax-m3": 1_000_000,
    "minimax-m2": 204_800,
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4": 1_048_576,
    "deepseek-v3": 131_072,
    "deepseek-chat": 131_072,
    "deepseek-reasoner": 131_072,
    "qwen3": 131_072,
    "qwen-max": 32_768,
    "glm-4": 131_072,
    "kimi-k2": 262_144,
    "moonshot-v1-128k": 131_072,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "claude-": 200_000,
}
_FALLBACK_WINDOW = 32_768   # 查不到时的保守值,宁可估小也别撞墙


def lookup_window(model: str) -> tuple[int, bool]:
    """按模型名查窗口,返回 (窗口, 是否查到)。查不到给保守值。"""
    # 上游常带 provider 前缀(硅基流动的 deepseek-ai/DeepSeek-V4-Flash),取斜杠后那段
    name = (model or "").strip().lower().rsplit("/", 1)[-1]
    hit = max((k for k in _KNOWN_WINDOWS if name.startswith(k)), key=len, default=None)
    return (_KNOWN_WINDOWS[hit], True) if hit else (_FALLBACK_WINDOW, False)


def resolve_window() -> int:
    """窗口取值优先级:配置显式填写 > 按模型名查表 > 保守默认。"""
    if settings.model_context_window > 0:
        return settings.model_context_window
    return lookup_window(settings.chat_model)[0]


@dataclass(frozen=True)
class ContextBudget:
    """一次请求的预算分解,除 turns 外单位都是 token。"""

    window: int             # 模型窗口
    fixed: int              # 固定开销:系统提示 + 证据 + 摘要 + 输出预留 + 余量
    sliding: int            # 滑窗可用
    turns: int              # 这个预算覆盖的轮数
    peak: int               # 当前这一轮的瞬时峰值,已从 room 里扣掉
    limited_by: str         # turns 轮数项生效 | window 窗口项生效
    healthy: bool           # 装不下一轮峰值或覆盖不足一轮为 False,启动自检据此报警

    @property
    def per_turn(self) -> int:
        """历史里一轮的稳态占用。不是「最坏情况」——最坏看 peak。"""
        return history_per_turn()


def summary_tokens() -> int:
    """摘要预留:注入几段就按几段算。"""
    return settings.summary_inject_segments * settings.summary_tokens_per_segment


def evidence_tokens(top_k: int | None = None) -> int:
    """证据预算:精排出几条就按几条算,调 rerank_top_k 预算跟着变。"""
    k = settings.rerank_top_k if top_k is None else top_k
    return k * settings.evidence_tokens_per_chunk


def turn_peak_tokens() -> int:
    """一轮 ReAct 的瞬时峰值,不含固定开销、不含历史。

    最多走 max_agent_steps 步,每步留一条 AI(带 tool_call)和一条工具结果在消息列表里
    喂给下一步,到本轮结束前都在窗口里。最后那条答复占的是 fixed 里那份输出预留——
    模型调用是串行的,Y 只需要留一份,不乘步数。"""
    return settings.max_user_input_tokens + settings.max_agent_steps * (
        settings.tool_result_max_tokens + settings.agent_step_ai_tokens)


def history_per_turn() -> int:
    """历史里一轮的稳态占用。

    工具结果不按原尺寸算:层 1 边界挪过去以后,compress_tool_result 把超过
    layer2_tool_keep_tokens 的压成一行。按原尺寸算是给一个不会出现的状态留位置。"""
    return (settings.max_user_input_tokens + settings.max_output_tokens
            + settings.max_agent_steps * settings.layer2_tool_keep_tokens)


def compute(window: int | None = None) -> ContextBudget:
    """按当前配置算一次预算。纯计算,不碰 IO。"""
    w = resolve_window() if window is None else window
    per_turn = history_per_turn()
    peak = turn_peak_tokens()

    fixed = (
        settings.system_prompt_tokens
        + evidence_tokens()
        + summary_tokens()
        + settings.max_output_tokens        # 输出预留不能小于 API 的 max_tokens
        + settings.budget_safety_margin
    )

    # 先把「当前这一轮跑起来要占的地方」扣掉,剩下的才轮到历史。
    # 不扣这一刀,32K 窗口下也会报健康,而那种窗口第 4 步就撞墙。
    room = w - fixed - peak
    if room <= 0:                           # 固定开销加本轮峰值就已经装不下
        return ContextBudget(w, fixed, 0, 0, peak, "window", False)

    by_turns = settings.context_budget_turns * per_turn
    sliding = min(by_turns, room)
    limited_by = "turns" if sliding == by_turns else "window"
    turns = sliding // per_turn
    return ContextBudget(w, fixed, sliding, turns, peak, limited_by, turns >= 1)


def describe(b: ContextBudget) -> str:
    """启动日志与自检用的一行说明。"""
    return (
        f"窗口={b.window} 固定开销={b.fixed} 单轮峰值={b.peak} 滑窗={b.sliding} "
        f"覆盖轮数={b.turns}(历史每轮 {b.per_turn}) 受限于={b.limited_by} "
        f"健康={'是' if b.healthy else '否'}"
    )
