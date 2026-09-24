import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
# 中文无词边界,分隔符优先段落/换行,再句末标点,最后逐字
CJK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", " ", ""]


def split_sections(md: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    return splitter.split_text(md)


def recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=CJK_SEPARATORS, is_separator_regex=False, length_function=len,
    )
    return splitter.split_text(text)


_SENT_RE = re.compile(r"[^。！？!?…\n]*[。！？!?…\n]|[^。！？!?…\n]+$")


def _split_sentences(text: str) -> list[str]:
    return [m for m in _SENT_RE.findall(text) if m]


def _trailing_sentences(text: str, max_chars: int) -> str:
    """取 text 结尾若干**完整句**作为重叠,总长尽量不超过 max_chars;
    单句超长则整句保留(优先「不留半截话」)。"""
    out: list[str] = []
    total = 0
    for s in reversed(_split_sentences(text)):
        if out and total + len(s) > max_chars:
            break
        out.insert(0, s)
        total += len(s)
    return "".join(out)


def apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]:
    if not chunks:
        return []
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        ov = _trailing_sentences(chunks[i - 1], overlap)
        out.append(ov + chunks[i] if ov else chunks[i])
    return out


_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _find_table_header(lines: list[str]) -> int:
    """返回表头行下标(其后紧跟 --- 分隔行);无表格返回 -1。
    允许表格前有若干散文行(不再要求首行即表头)。"""
    for i in range(len(lines) - 1):
        if (lines[i].lstrip().startswith("|")
                and _TABLE_SEP_RE.match(lines[i + 1]) and "-" in lines[i + 1]):
            return i
    return -1


def is_table_block(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return _find_table_header(lines) != -1


def split_table_rows(table_md: str, max_rows: int) -> list[str]:
    """大表格按行切,每块重贴表头行 + 分隔行;不超 max_rows 数据行则整块返回。
    表头前的散文行作为前言,保留在首块(不丢),后续块只带表头。"""
    lines = [ln for ln in table_md.strip().splitlines() if ln.strip()]
    idx = _find_table_header(lines)
    if idx == -1:
        return [table_md.strip()]
    preamble, header, sep, rows = lines[:idx], lines[idx], lines[idx + 1], lines[idx + 2:]
    if len(rows) <= max_rows:
        return [table_md.strip()]
    out: list[str] = []
    for j, i in enumerate(range(0, len(rows), max_rows)):
        group = rows[i:i + max_rows]
        block = [*preamble, header, sep, *group] if j == 0 else [header, sep, *group]
        out.append("\n".join(block))
    return out
