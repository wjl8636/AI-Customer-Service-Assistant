# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 飞轮批处理:扫问题池未归并条目 → 标准化+查重 → 待审队列。
运行:make flywheel(需 mysql + 聊天上游在线)。定时跑给 cron 示例:
  */30 * * * * cd /path/to/mewhelp && make flywheel >> log/flywheel.log 2>&1
"""
import asyncio
import sys

from app.core.flywheel import process_pending


async def main():
    stats = await process_pending(limit=200)
    print(f"本轮处理 {stats['processed']} 条:新建缺口 {stats['created']},"
          f"归并 {stats['merged']},跳过待重试 {stats['skipped']}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
