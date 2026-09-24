from app.core.llm import get_chat_model
from app.core.prompts import COREF_REWRITE_PROMPT


async def resolve(query: str, history: str = "") -> str:
    """指代消解 + 口语归一。结合历史把半截话补成完整问句;已完整则原样。
    出错兜底:返回原句(不阻断下游)。"""
    try:
        model = get_chat_model()
        r = await (COREF_REWRITE_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"})
        text = (r.content if isinstance(r.content, str) else "").strip()
    except Exception:
        return query
    return text or query
