# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import logging

from pydantic import BaseModel, Field

from app.core import llm
from app.core.llm import get_chat_model
from app.core.prompts import SELF_CHECK_PROMPT

logger = logging.getLogger(__name__)


class _Check(BaseModel):
    useful: bool = Field(description="证据是否足以回答")
    reason: str = Field(default="", description="判断依据")


def _chain():
    return SELF_CHECK_PROMPT | llm.structured(_Check)


async def check_sufficient(query: str, evidence_texts: list[str]) -> dict:
    """生成前证据充分性自评。扁平字段 useful/reason(避开 glm-5.2 嵌套数组 502)。

    模型异常按「证据不足」处理,不按「够用」放行。这道闸的职责就是拦住证据撑不住的回答,
    它自己挂掉时无从判断够不够,放行等于把没过闸的内容直接送去生成,可能编出一段假政策。
    判不足最多是让用户看到兜底话术、多走一次人工,代价小得多。这类问题会进飞轮池,
    人工复审时看到证据其实够、却记着 self_check 兜底,正好是这道闸出过故障的线索。"""
    evidence = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_texts)) or "(无证据)"
    try:
        r: _Check = await _chain().ainvoke({"query": query, "evidence": evidence})
    except Exception:
        logger.warning("证据自评失败,按证据不足处理 query=%r", query, exc_info=True)
        return {"useful": False, "reason": "证据自评失败"}
    return {"useful": bool(r.useful), "reason": r.reason or ""}
