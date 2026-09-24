"""验收展示(dry-run):列出建库材料 + 预览结构化切块结果,不写库、不碰 Milvus、不调上游。
展示需求1 的三个切块特性:按标题层级切 / 表格按行拆(表头复制)/ 句子边界 overlap。
运行:PYTHONPATH=. uv run python scripts/show_kb.py"""
import pathlib

from app.kb import chunking, documents
from app.kb.sources import KB_DIR, SOURCE_TYPES as DOCS

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED_SQL = ROOT / "sql" / "ch03-seed.sql"


def _preview(text: str, n: int = 46) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[:n] + "…"


def main() -> None:
    print("=== 离线建库材料清单 ===")
    print("文档(结构化切块 → knowledge_chunks):")
    for fname, ctype in DOCS.items():
        p = KB_DIR / fname
        raw = p.read_text(encoding="utf-8")
        print(f"  data/kb/{fname:<24} [{ctype:<6}] {len(raw):>4} 字符 / {raw.count(chr(10)) + 1} 行")
    print("历史会话(挖知识 job 的【输入】,抽 QA → knowledge_chunks):")
    print(f"  {SEED_SQL.relative_to(ROOT)} → seed-u1/u2 共 2 会话、4 消息:")
    print("    · u1「你们发货一般多久啊」/「现货…48 小时内发货」 → 可抽:发货时效")
    print("    · u2「满多少包邮」/「满 99 元包邮,未满 10 元运费」 → 可抽:运费/包邮")
    print("  ⚠ 挖掘要调 LLM(glm-5.2),不属于本地 dry-run;抽取结果见 make eval-mining / make kb-mine")

    print("\n=== 切块预览(dry-run,不写库)===")
    tally: dict[str, int] = {}
    key_total = 0
    table_notes: list[str] = []
    for fname, ctype in DOCS.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        tally[ctype] = tally.get(ctype, 0) + len(chunks)   # 同一 content_type 可能有多份材料,要累加不能覆盖
        # 按 section_path 分组,标出被拆成多块的节(表格/超长散文)
        by_path: dict[str, list[documents.Chunk]] = {}
        for c in chunks:
            by_path.setdefault(c.section_path, []).append(c)
        print(f"\n▼ {fname} [{ctype}]  →  {len(chunks)} 块")
        for path, group in by_path.items():
            multi = f"  (该节切成 {len(group)} 块)" if len(group) > 1 else ""
            print(f"  ┌ 节:{path}{multi}")
            for j, c in enumerate(group, 1):
                key_total += c.is_key_clause
                flag = " ★关键条款" if c.is_key_clause else ""
                seq = f"{j}/{len(group)}" if len(group) > 1 else "-"
                is_tbl = chunking.is_table_block(c.answer)
                kind = "表格块" if is_tbl else "文本块"
                print(f"  │ [{kind} {seq}]{flag}  Q(questions)={c.questions}  category={c.category}")
                print(f"  │   A({len(c.answer)}字): {_preview(c.answer)}")
                if is_tbl and len(group) > 1:
                    header = c.answer.strip().splitlines()[0]
                    table_notes.append(f"{fname}「{path.split(' / ')[-1]}」块{seq} 复制表头: {header.strip()}")
    print("\n=== 汇总 ===")
    print(f"总块数 {sum(tally.values())};类型分布 " + " ".join(f"{k}={v}" for k, v in tally.items()))
    print(f"关键条款(is_key_clause=1)共 {key_total} 块(命中 运费/邮费/退款/保修 等词)")
    if table_notes:
        print("表格按行拆(表头复制)已触发:")
        for n in table_notes:
            print(f"  · {n}")
    print("句子边界 overlap:仅当单节 > chunk_size(400字)被 recursive_split 成多块时触发;"
          "当前各节均较短未触发,该逻辑由 tests/test_chunking.py 锁定。")


if __name__ == "__main__":
    main()
