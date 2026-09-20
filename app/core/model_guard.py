# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""型号机械闸:答案里出现的型号必须逐字来自本轮证据,否则这句话是编的。

ch04 评估里判出的唯一一条真幻觉是型号:库里是 MH-CAM1,答案写成了 MH-CAD1——
一个不存在的型号。这类错误的特点是**机械可判**:型号是封闭集合,答案里的每个型号
串要么在证据里出现过,要么就是模型自己拼的。判它不需要再调一次大模型,正则就够,
所以别把这件事交给忠实度裁判(裁判贵、还会漏)。

和 ch09 读图小注里那道数字校验是同一手法:**能机械校验的,就不要用模型校验。**

只认「MH-字母-数字」这一种编号形态(MH-LP100 / MH-W40 / MH-CAM1),这是本项目商品
型号的写法。大小写不敏感,但比对时按原样比——MH-cam1 这种大小写不一致也当没命中,
因为客服答案里的型号本该是从证据里复制过去的。
"""
import re

MODEL_RE = re.compile(r"MH-[A-Za-z]{1,4}\d{1,4}")


def models_in(text: str) -> list[str]:
    """按出现顺序去重列出文本里的型号串。"""
    seen: dict[str, None] = {}
    for m in MODEL_RE.findall(text or ""):
        seen.setdefault(m, None)
    return list(seen)


def unsupported_models(answer: str, evidence: str) -> list[str]:
    """答案里有、证据里没有的型号。空列表 = 这一关过了。"""
    allowed = set(models_in(evidence))
    return [m for m in models_in(answer) if m not in allowed]


def repair_hint(bad: list[str]) -> str:
    """重写提示:只告诉它哪几个型号没依据,不替它猜正确型号——猜是幻觉的来源。"""
    return (
        "上一版回答里这些型号在给你的证据里找不到:" + "、".join(bad) + "。"
        "请重写回答:型号必须逐字复制证据里出现过的型号串,证据里没有的型号一个都不要写,"
        "拿不准就不提型号。其余内容与引用编号保持不变。"
    )
