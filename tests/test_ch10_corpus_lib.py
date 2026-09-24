"""ch10 语料纯函数:脱敏 / 去重 / 分层划分。"""
from scripts.ch10.corpus_lib import dedupe, desensitize, split_dataset


def test_desensitize_masks_sensitive_keeps_model_no():
    assert desensitize("我手机13812345678帮我查下") == "我手机[手机号]帮我查下"
    assert desensitize("订单202601180001234567还没发") == "订单[单号]还没发"
    assert desensitize("加我微信 cat_lover2026 聊") == "加我微信[账号]聊"
    # 商品型号不许误伤
    assert "MH-LP100" in desensitize("MH-LP100 猫砂盆废砂盒多久倒一次")


def test_dedupe_keeps_first():
    out = dedupe([{"text": "a", "origin": "pool"}, {"text": "a", "origin": "simulated"},
                  {"text": "b", "origin": "pool"}])
    assert [s["text"] for s in out] == ["a", "b"]
    assert out[0]["origin"] == "pool"


def _make_samples():
    samples = []
    for cls, n in (("退换货", 40), ("物流", 40), ("尺码", 30)):
        samples += [{"text": f"{cls}问题{i}", "labels": [cls]} for i in range(n)]
    samples += [{"text": f"多诉求{i}", "labels": ["尺码", "退换货"]} for i in range(12)]
    samples += [{"text": f"小组合{i}", "labels": ["物流", "退换货"]} for i in range(3)]
    return samples


def test_split_ratio_and_no_leakage():
    train, val, test = split_dataset(_make_samples())
    total = len(train) + len(val) + len(test)
    assert total == 125
    assert 0.72 <= len(train) / total <= 0.88
    texts = [s["text"] for s in train + val + test]
    assert len(texts) == len(set(texts))


def test_split_every_class_in_every_split():
    train, val, test = split_dataset(_make_samples())
    for split in (train, val, test):
        found = {lb for s in split for lb in s["labels"]}
        assert {"退换货", "物流", "尺码"} <= found


def test_split_deterministic():
    a = split_dataset(_make_samples(), seed=42)
    b = split_dataset(_make_samples(), seed=42)
    assert a == b
