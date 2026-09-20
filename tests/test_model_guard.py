# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""型号机械闸:拿 ch04 评估真判出来的那条幻觉当第一条用例。"""
from app.core import model_guard as g

# A52 的真实答案片段:库里只有 MH-CAM1 和 MH-FD30,答案里冒出了不存在的 MH-CAD1
EVIDENCE = ("[1] 全景看护摄像头(型号 MH-CAM1): 供电:Type-C 供电,需连接 2.4G Wi-Fi\n"
            "[2] 自动喂食器 摄像头版(型号 MH-FD30): 粮桶容量:6L")
ANSWER_BAD = "确认 Type-C 供电线连接正常(针对 MH-CAD1 型号)[1],摄像头版(MH-FD30)保修 12 个月[2]。"
ANSWER_OK = "确认 Type-C 供电线连接正常(针对 MH-CAM1 型号)[1]。"


def test_揪出证据里没有的型号():
    assert g.unsupported_models(ANSWER_BAD, EVIDENCE) == ["MH-CAD1"]


def test_型号都来自证据时放行():
    assert g.unsupported_models(ANSWER_OK, EVIDENCE) == []


def test_没提型号也放行():
    assert g.unsupported_models("满 99 元包邮[1]。", EVIDENCE) == []


def test_多个型号按出现顺序去重():
    bad = g.unsupported_models("MH-XX9 与 MH-YY1,再提一次 MH-XX9", EVIDENCE)
    assert bad == ["MH-XX9", "MH-YY1"]


def test_大小写不一致也算没命中():
    # 型号本该是从证据里复制过去的,大小写都对不上说明是模型自己写的
    assert g.unsupported_models("针对 mh-cam1 型号", EVIDENCE) == []      # 不匹配型号形态,不误报
    assert g.unsupported_models("针对 MH-Cam1 型号", EVIDENCE) == ["MH-Cam1"]


def test_不把普通编号当型号():
    for s in ("[1]", "2.4G Wi-Fi", "128G TF 卡", "MH-", "MHLP100"):
        assert g.models_in(s) == [], s


def test_重写提示只点名不猜正确型号():
    hint = g.repair_hint(["MH-CAD1"])
    assert "MH-CAD1" in hint and "MH-CAM1" not in hint
