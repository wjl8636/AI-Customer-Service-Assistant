# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from app.core.taxonomy import (ID2LABEL, LABEL2ID, NUM_CLASSES, TOPIC_CLASSES,
                               TOPIC_NAMES, terminology_table)


def test_seventeen_unique_classes():
    assert NUM_CLASSES == 17
    assert len(set(TOPIC_NAMES)) == 17


def test_four_leads_first():
    assert TOPIC_NAMES[:4] == ("退换货", "物流", "尺码", "发票")


def test_every_class_has_boundary_examples_severity():
    for c in TOPIC_CLASSES:
        assert c.boundary.strip()
        assert len(c.examples) >= 3 or c.name == "其他"
        assert c.severity in ("严", "中", "宽")


def test_label_id_roundtrip():
    for name, i in LABEL2ID.items():
        assert ID2LABEL[i] == name
    assert LABEL2ID["退换货"] == 0 and LABEL2ID["其他"] == 16


def test_terminology_table_covers_all():
    table = terminology_table()
    for name in TOPIC_NAMES:
        assert name in table
