# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import re

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)  # \W 不含中文(中文属 \w),故只去空白/标点


def normalize_question(q: str) -> str:
    return _STRIP_RE.sub("", q.strip().lower())


def dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]:
    """去重:staging 内部 + 对已有 knowledge 问法;返回 (kept, discarded)。"""
    seen = {normalize_question(q) for q in existing_questions}
    kept, discarded = [], []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            discarded.append(item)
        else:
            seen.add(key)
            kept.append(item)
    return kept, discarded
