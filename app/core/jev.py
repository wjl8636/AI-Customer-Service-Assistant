"""typesafe.ai 的 Jev——System One 决策模型客户端(原生 HTTP,不经 SDK)。

Jev 不走文本生成,而是提交一份 `state` + 一组「类型化问题」(Choice/Score/Noul),
直接返回结构化答案。本项目用它在意图识别上做主分类器:Choice 原语从九类里选一,
返回选中项 + 每项概率 + 置信度,并随响应带回 token 用量。

端点:POST {jev_base_url}/systemone  (jev_base_url 默认 https://api.typesafe.ai/v1)
鉴权:Bearer <jev_api_key>

失败统一抛 JevError(可读中文),由调用点降级成安全默认值——不在客户端降级,以免把
「上游抽风」和「真没配 key」混成同一种静默结果。
"""
from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class JevError(RuntimeError):
    """Jev 调用失败。message 为可读中文,供调用点记录或兜底。"""


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.jev_api_key}",
        "Content-Type": "application/json",
    }


async def choice(state: str, instructions: str, criteria: dict,
                 question_id: str = "intent") -> dict:
    """向 Jev 发一个 Choice 问题(九选一),返回:

    {
      "choice": str,            # 选中的选项(就是 criteria 的某个 key)
      "probabilities": dict,    # 每个选项的概率(和为 1)
      "confidence": float,      # 模型对该判断的把握 0-1
      "input_tokens": int,      # 本次调用输入 token
      "output_tokens": int,     # 本次调用输出 token
    }

    criteria: {选项: 该选项的边界说明};选项即答案。string 就足够,不必传结构化 criteria。
    失败抛 JevError。调用点负责 catch 并按意图取「其他」兜底。
    """
    if not settings.jev_api_key:
        raise JevError("JEV_API_KEY 未配置:无法调用 Jev,请检查 .env")

    url = settings.jev_base_url.rstrip("/") + "/systemone"
    payload = {
        "state": state,
        "model": settings.jev_model,
        "questions": {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=_headers())
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise JevError(f"Jev HTTP {e.response.status_code}: {e.response.text[:200] or e}") from e
    except httpx.HTTPError as e:
        raise JevError(f"Jev 网络错误: {e}") from e
    except ValueError as e:
        raise JevError(f"Jev 响应不是合法 JSON: {e}") from e

    ans = (data.get("answers") or {}).get(question_id) or {}
    usage = data.get("usage") or {}
    return {
        "choice": ans.get("choice", ""),
        "probabilities": ans.get("probabilities", {}),
        "confidence": float(ans.get("confidence", 0.0)),
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
    }