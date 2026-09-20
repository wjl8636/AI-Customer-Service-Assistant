# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""读图小注:模型可以写,但编出来的数一个都不许上页面——数字校验是这层的主闸。"""
import pytest

from app.core import read_notes

PAYLOAD = {
    "rows": [{"intent": "商品咨询", "count": 3, "tokens": 8821, "avg_tokens": 2940, "share": 0.47},
             {"intent": "物流", "count": 1, "tokens": 7942, "avg_tokens": 7942, "share": 0.42}],
    "total_tokens": 18720,
}


class _Stub:
    """假模型:回固定文本,记录被问了几次。"""

    def __init__(self, text, boom=False):
        self.text, self.boom, self.calls = text, boom, 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if self.boom:
            raise RuntimeError("上游没起")
        return type("R", (), {"content": self.text})()


def test_numbers_in_the_data_pass():
    """原样、四舍五入、百分比、千分位逗号都算同一个数。"""
    assert read_notes.verify("商品咨询占 47%,单均 2,940 token", PAYLOAD)
    assert read_notes.verify("总账 18720 token,物流单均 7942", PAYLOAD)


def test_ratio_complement_passes():
    """0.85 的通过率说成「15% 被误拦」是同一个数的另一面,不算编。"""
    assert read_notes.verify("代价是 15% 的可答被误拦", {"pass_rate": 0.85})


def test_numbers_inside_identifiers_are_not_claims():
    """p25、Recall@10、bge-m3 里的数字是名字的一部分,不是模型下的结论,不能因此毙掉整条注。"""
    assert read_notes.verify("可答的 p25 已经到 0.47,Recall@10 那栏别看了", PAYLOAD)
    assert read_notes.verify("重排用的是 bge-reranker-v2-m3,单均 7942 最高", PAYLOAD)


def test_invented_number_is_rejected():
    """数据里没有 61%,这条注整条不要——页面显示旧口径的句子也比印错数好。"""
    assert read_notes.verify("商品咨询占 61%", PAYLOAD) is False


async def test_generate_returns_verified_text():
    stub = _Stub("商品咨询占 47%,要瘦成本先动它")
    note = await read_notes.generate("cost_by_intent", PAYLOAD, model=stub)
    assert note == "商品咨询占 47%,要瘦成本先动它。"
    assert stub.calls == 1


@pytest.mark.parametrize("raw,want", [
    # 全角逗号与分号 → 半角,逗号后不留空格:跟页面上兜底那几句同一个口径
    ("商品咨询占 47%，先给它瘦 prompt；物流那路再说", "商品咨询占 47%,先给它瘦 prompt,物流那路再说。"),
    ("占比 0.47, 单均 2940 最高", "占比 0.47,单均 2940 最高。"),
    # 数字紧贴汉字读起来挤,页面上那几句都是留一格的
    ("口语类才0.85,答题要点漏了", "口语类才 0.85,答题要点漏了。"),
    ("0.85的通过率够用", "0.85 的通过率够用。"),
    # 句末只留一个句号,多出来的叹号问号一律削掉
    ("要点覆盖全。!", "要点覆盖全。"),
    ("先看这条线", "先看这条线。"),
    # 汉字与英文之间也留一格,粘在英文词后面的数字要拆出来才校验得到
    ("商品咨询总token8821,先瘦prompt", "商品咨询总 token 8821,先瘦 prompt。"),
    # 名字里的数字不能被拆走:p25、bge-m3、Recall@10 拆了就不是那个名字了
    ("可答 p25 到 0.47,重排用 bge-m3,Recall@10 别看", "可答 p25 到 0.47,重排用 bge-m3,Recall@10 别看。"),
])
def test_tidy_matches_the_pages_own_typography(raw, want):
    """排版归一是机械活,不指望 prompt 稳:注和兜底句同处一行,笔法不能看出两种。"""
    assert read_notes.tidy(raw) == want


async def test_generated_note_comes_back_tidied():
    stub = _Stub("商品咨询占 47%，先给它瘦 prompt")
    assert await read_notes.generate("cost_by_intent", PAYLOAD, model=stub) \
        == "商品咨询占 47%,先给它瘦 prompt。"


async def test_generate_drops_hallucinated_note():
    stub = _Stub("商品咨询占 61%,已经烧掉 99,999 token")
    assert await read_notes.generate("cost_by_intent", PAYLOAD, model=stub) is None


async def test_generate_drops_overlong_note():
    """读图是一句话,不是又一段正文。"""
    stub = _Stub("商品咨询占 47%。" * 40)
    assert await read_notes.generate("cost_by_intent", PAYLOAD, model=stub) is None


async def test_upstream_down_is_not_an_error():
    """上游没起只是这一轮没有注:产物照样落盘,页面回落兜底句。"""
    stub = _Stub("", boom=True)
    assert await read_notes.generate("cost_by_intent", PAYLOAD, model=stub) is None


async def test_generate_all_keeps_only_the_good_ones():
    class _PerKind:
        async def ainvoke(self, prompt):
            text = "物流单均 7,942 token 最高" if "多步工具链" in prompt else "凭空的 12345 条"
            return type("R", (), {"content": text})()

    notes = await read_notes.generate_all(
        {"cost_by_intent": PAYLOAD, "rag_mrr": PAYLOAD}, model=_PerKind())
    assert list(notes) == ["cost_by_intent"]


@pytest.mark.parametrize("kind", sorted(read_notes.KINDS))
def test_every_kind_declares_what_and_why(kind):
    """每类图都得交代「画的是什么」和「读者要做什么判断」,不然模型只能瞎猜。"""
    what, decision = read_notes.KINDS[kind]
    assert len(what) > 10 and len(decision) > 8
