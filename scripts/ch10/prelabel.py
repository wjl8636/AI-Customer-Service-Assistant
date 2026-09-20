# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch10 预标:大模型照权威术语表给问题打多标签。折中路线的前半——预标,后半人工抽审。"""
import asyncio

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.taxonomy import LABEL2ID, terminology_table


class _Labeled(BaseModel):
    labels: list[str] = Field(description="命中的类目名数组,取术语表类目名原文")


PRELABEL_PROMPT = """你是电商客服主题标注员。对照下面这份 17 类权威归并术语表,给用户问题打主题标签。

术语表(类目:边界说明(示例)):
{terminology}

标注铁律:
1. 字面提到几个诉求就打几个标签,一个不多一个不少。「买大了想退」→ 尺码+退换货;\
「这鞋买大了一码」→ 只标尺码,不许因为「可能要退」就脑补退换货。
2. 近邻边界:修归保修维修、退归退换货;运费管钱、物流管货;价保是补差价、优惠活动是券和满减。
3. 方言、口语、错别字要看穿字面认诉求:「俺买的那玩意儿咋还没到俺这疙瘩」是物流;「退活」是「退货」的错字。
4. 都对不上就标「其他」;标签必须取术语表的类目名原文。

用户问题:{text}"""


async def prelabel_one(text: str) -> list[str]:
    model = get_chat_model().with_structured_output(_Labeled)
    try:
        r: _Labeled = await model.ainvoke(
            PRELABEL_PROMPT.format(terminology=terminology_table(), text=text))
    except Exception as e:
        # 失败要看得见:无声标「其他」会污染语料,还会把黄金样例闸的通过率无声拉低
        print(f"[prelabel] 调用失败,兜底标「其他」: {text[:30]}… ({type(e).__name__}: {e})")
        return ["其他"]
    labels = [lb for lb in r.labels if lb in LABEL2ID]
    return labels or ["其他"]


async def prelabel_batch(texts: list[str], concurrency: int = 8) -> list[list[str]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(t: str) -> list[str]:
        async with sem:
            return await prelabel_one(t)

    return list(await asyncio.gather(*[one(t) for t in texts]))
