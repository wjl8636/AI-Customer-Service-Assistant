"""改完 data/kb/*.md 之后,只把改动的那几块重新入库(不整库重建)。

为什么需要这个脚本:补库是评估驱动优化的常规动作——评估判出一条编造,回头看是库里
少写了一格,补上再跑。但整库重建(make kb-reset && kb-build && kb-vectorize)会把
**飞轮审核通过写回的块一起清掉**,那些块只在库里、文件里没有,清了就找不回来。

所以这里按 section_path 对齐「文件里的块」与「库里的块」,逐块比正文:
  - 正文变了 → 改写库里那一块并退回 pending(Milvus 那边主键就是 chunk id,重嵌=原地更新);
  - 文件里多出来的小节 → 只报告不自动插,新增小节该走 make kb-build 或录入页;
  - 库里有、文件里没有的 → 只报告不删,飞轮写回的块就长这样,删了就出事。

改完记得 make kb-vectorize 把 pending 的重嵌进 Milvus。
运行:make kb-repatch
"""
import asyncio
import sys

from app.db import repository
from app.kb import documents
from app.kb.sources import KB_DIR, SOURCE_TYPES


def _norm(s: str) -> str:
    return "\n".join(line.rstrip() for line in (s or "").strip().splitlines())


async def main() -> int:
    in_db = await repository.list_chunks_by_content_types(set(SOURCE_TYPES.values()))
    # 一个 section_path 可能对应多块:长表格会被切成好几块,section_path 是一样的。
    # 所以按「路径 + 第几块」对齐,不能只按路径——按路径存 dict 会让后一块覆盖前一块,
    # 再拿文件里的第一块去更新它,等于把后半张表原地改成了前半张(踩过这个坑)。
    by_path: dict[str, list] = {}
    for r in in_db:                                   # list_chunks_by_content_types 已按 id 排序
        by_path.setdefault(r.section_path, []).append(r)
    changed = added = 0
    used: dict[str, int] = {}                         # 每个路径已经对上了几块(跨文件累计)

    for fname, ctype in SOURCE_TYPES.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        for c in documents.build_chunks(md, content_type=ctype):
            i = used.get(c.section_path, 0)
            used[c.section_path] = i + 1
            rows = by_path.get(c.section_path) or []
            row = rows[i] if i < len(rows) else None
            if row is None:
                print(f"  + 文件里有、库里没有(本脚本不插,走 kb-build / 录入页):"
                      f"{c.section_path} 第 {i + 1} 块")
                added += 1
                continue
            if _norm(row.answer) == _norm(c.answer) and _norm(row.questions) == _norm(c.questions):
                continue
            await repository.repend_chunk_text(row.id, c.questions, c.answer)
            print(f"  ~ 正文已更新(id={row.id}):{c.section_path}")
            changed += 1

    for path, rows in by_path.items():
        for row in rows[used.get(path, 0):]:
            print(f"  ! 库里有、文件里没有(不动它,可能是飞轮写回的):id={row.id} {path}")

    pending = await repository.count_chunks_by_status("pending")
    print(f"\n改动 {changed} 块 · 新小节 {added} 个 · 当前 pending {pending} 块")
    if pending:
        print("下一步:make kb-vectorize(把 pending 的重嵌进 Milvus)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
