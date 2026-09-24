import time
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.core import jev
from app.core import llm
from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他")


class _Intent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"] = Field(
        description="九类意图之一")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="判断把握 0-1")


# ---- Jev Choice 的九选一定义(与 INTENT_CLASSIFY_SYSTEM 同一套边界,保证两个后端口径一致)----
INTENT_CHOICE_INSTRUCTION = "判断这条电商客服用户消息的主要意图,九选一(输入已是指代消解后的完整问题,不要再改写)。"

INTENT_CHOICE_CRITERIA: dict[str, str] = {
    "物流": "查快递到哪了、发货没有,通常带订单号/单号",
    "订单": "查某订单的状态、金额、下单时间、买了什么",
    "商品咨询": "商品参数、退换货政策、通用FAQ这类查规则查说明的问题(未锁定到某一单)",
    "退款退货": "要对某个已购订单退货或退款(得先查这一单能不能退)",
    "售后": "维修、保修、换新这类售后处理(不含退款退货)",
    "投诉": "对产品或服务不满、要追责、要说法",
    "人工": "明确要求建工单、转人工、找客服专员跟进(无论是否带着具体问题)",
    "闲聊": "寒暄、玩笑、与购物无关的话题",
    "其他": "拿不准、又不该硬塞进上面某类时选它(兜底)",
}


def _jev_state(query: str, history: str) -> str:
    return f"最近对话(可空):\n{history or '(无)'}\n\n当前用户这句话:{query}"


def _pick(intent: str) -> str:
    """越界/未知一律归「其他」(最保守兜底)。两个后端共用。"""
    return intent if intent in INTENTS else "其他"


# ---- 后端一:原 Chat 模型结构化识别(默认,行为不变)----
async def _llm_classify(query: str, history: str = "") -> dict:
    model = llm.structured(_Intent, slot="intent")
    try:
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"})
    except Exception:
        return {"intent": "其他", "confidence": 0.0}
    return {"intent": _pick(r.intent), "confidence": float(r.confidence)}


# ---- 后端二:Jev 决策模型 Choice 原语----
async def _jev_classify(query: str, history: str = "") -> dict:
    try:
        r = await jev.choice(_jev_state(query, history), INTENT_CHOICE_INSTRUCTION,
                             INTENT_CHOICE_CRITERIA)
    except Exception:
        return {"intent": "其他", "confidence": 0.0}
    return {"intent": _pick(r["choice"]), "confidence": r["confidence"]}


async def classify(query: str, history: str = "") -> dict:
    """九类意图 + confidence(ch08 增「人工」:明确要求建工单/转人工)。

    实现经 settings.intent_backend 切换:llm=原 Chat 模型结构化识别(默认,行为不变);
    jev=Jev 决策模型的 Choice 原语。解析失败/越界 → 归「其他」(最保守兜底)。
    返回扁平 {intent, confidence}——上游节点只依赖这两个字段,换后端不影响它们。"""
    if settings.intent_backend == "jev":
        return await _jev_classify(query, history)
    return await _llm_classify(query, history)


async def _raw_llm_usage(query: str, history: str) -> tuple[int, int]:
    """同一 prompt 走一次原始模型调用,读 usage_metadata。
    结构化输出的 usage 拿不到(raw 才暴露),所以 token 另测一次、不计入耗时。
    thinking=False 与生产 structured() 一致;失败返回 (0,0)。"""
    try:
        msgs = INTENT_CLASSIFY_PROMPT.format_messages(query=query, history=history or "(无)")
        m = get_chat_model(streaming=False, slot="intent", thinking=False)
        ai = await m.ainvoke(msgs)
        u = ai.usage_metadata or {}
        return (int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0))
    except Exception:
        return (0, 0)


async def classify_with_usage(query: str, history: str = "") -> dict:
    """同 classify,但额外返回 token 用量与耗时——供意图后端测评
    (scripts/compare_intent_backends.py)使用,生产链路不调用它。

    返回 {intent, confidence, backend, latency_ms, input_tokens, output_tokens}:
      - Jev 路:一次真实调用,token 取自响应 usage(与耗时同一次)。
      - LLM 路:结构化识别判意图(生产同款);token 另走一次同 prompt 原始调用(不叠加进耗时)。
    """
    backend = settings.intent_backend
    t0 = time.perf_counter()
    if backend == "jev":
        try:
            r = await jev.choice(_jev_state(query, history), INTENT_CHOICE_INSTRUCTION,
                                 INTENT_CHOICE_CRITERIA)
        except Exception:
            r = {"choice": "", "confidence": 0.0, "input_tokens": 0, "output_tokens": 0}
        return {"intent": _pick(r["choice"]), "confidence": r["confidence"], "backend": "jev",
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}
    # LLM 路
    model = llm.structured(_Intent, slot="intent")
    try:
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"})
        intent, conf = _pick(r.intent), float(r.confidence)
    except Exception:
        intent, conf = "其他", 0.0
    itok, otok = await _raw_llm_usage(query, history)
    return {"intent": intent, "confidence": conf, "backend": "llm",
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "input_tokens": itok, "output_tokens": otok}