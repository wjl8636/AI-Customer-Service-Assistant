from app.kb import chunking


def test_split_sections_keeps_header_path():
    md = "# 售后手册\n\n## 退货政策\n\n支持7天无理由。\n\n## 运费说明\n\n满99包邮。"
    secs = chunking.split_sections(md)
    freight = [d for d in secs if d.metadata.get("h2") == "运费说明"]
    assert freight, "应切出「运费说明」小节"
    assert freight[0].metadata.get("h1") == "售后手册"
    assert "满99包邮" in freight[0].page_content


def test_recursive_split_breaks_oversized():
    text = "句子内容。" * 60  # 300 字
    parts = chunking.recursive_split(text, chunk_size=50, chunk_overlap=0)
    assert len(parts) > 1
    assert max(len(p) for p in parts) <= 50
