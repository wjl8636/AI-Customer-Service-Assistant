# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""裁判回归的取样与判定:台账里哪些个案能当考题、人工处置怎么翻译成标准答案。

这一层必须是纯函数、必须有测试:整轮回归的结论(裁判和人工一致率)全建在它上面。
角标原文拼回去的格式要和评估时喂给裁判的逐字一致——差一个空格,重放的就不是当时那道题了。
不调上游、不进库。
"""
from scripts.judge_check import (evidence_from_citations, expected_faithful,
                                 gradable, verdict_row)

CITS = [{"n": 1, "question": "退款时效", "answer": "到账时效以平台售后规则为准。"},
        {"n": 2, "question": "处理时限", "answer": "退款 3 个工作日。"}]


class _Row:
    def __init__(self, eval_id, status, citations):
        self.eval_id, self.status, self.citations = eval_id, status, citations


def test_角标原文拼回评估时那份证据():
    # 格式与 eval_ch04._format_evidence 逐字一致:[n] 问题:答案,一条一行
    assert evidence_from_citations(CITS) == (
        "[1] 退款时效:到账时效以平台售后规则为准。\n[2] 处理时限:退款 3 个工作日。")


def test_没记_n_的老快照按顺序补角标():
    assert evidence_from_citations([{"question": "运费", "answer": "买家承担。"}]) == "[1] 运费:买家承担。"
    assert evidence_from_citations(None) == ""


def test_人工处置翻译成裁判该给的答案():
    assert expected_faithful("已解决") is False      # 人工确认真编了 → 裁判该判 false
    assert expected_faithful("无需解决") is True     # 人工看过是判严了 → 裁判该判 true
    assert expected_faithful("未解决") is None       # 还没过目,没有标准答案


def test_能考的个案要有处置也要有快照():
    rows = [_Row("A50", "无需解决", CITS),          # 有处置有快照 → 能考
            _Row("E15", "已解决", CITS),
            _Row("E17", "未解决", CITS),            # 没过目,不知道对错
            _Row("A43", "已解决", None)]            # 老个案没留快照,重放不出来
    assert [r.eval_id for r in gradable(rows)] == ["A50", "E15"]


def test_一致与不一致():
    assert verdict_row("E15", "已解决", False)["agree"] is True
    assert verdict_row("E15", "已解决", True)["agree"] is False      # 真编造被放过
    assert verdict_row("A50", "无需解决", False)["agree"] is False   # 判严了
    # 调用失败不能算一致——否则上游一抖,一致率就虚高
    row = verdict_row("A50", "无需解决", None)
    assert row["agree"] is False and row["actual"] is None
