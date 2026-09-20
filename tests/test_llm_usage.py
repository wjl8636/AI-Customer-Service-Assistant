# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""流式用量不许被重复累加。

为什么要有这组测试:有些 OpenAI 兼容通道每个 chunk 都重报一遍累计用量,而 langchain-openai
把各 chunk 相加,一次调用的 token 数被放大到 chunk 条数倍。ch09 的成本账直接读这个数,
虚高一个量级;判断口径一旦按它做闸,还会误伤(ch05 早先那道预算闸就是这么第一步跳闸的)。
不打上游,拿假 chunk 喂转换入口。
"""
from langchain_core.messages import AIMessageChunk

from app.core.llm import get_chat_model


def _feed(model, usages: list[dict]) -> dict:
    """把一串 chunk 喂进转换入口,按库的方式相加,返回聚合后的 usage_metadata。"""
    acc = None
    for u in usages:
        gc = model._convert_chunk_to_generation_chunk(
            {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
             "usage": u},
            AIMessageChunk, None)
        acc = gc.message if acc is None else acc + gc.message
    return acc.usage_metadata


def _cumulative(n: int, prompt: int) -> list[dict]:
    """模拟每个 chunk 都重报累计用量的通道:input 恒定,output 逐个涨。"""
    return [{"prompt_tokens": prompt, "completion_tokens": i,
             "total_tokens": prompt + i} for i in range(n)]


def test_每个chunk都重报累计用量时_聚合出来仍是真实用量():
    m = get_chat_model()
    um = _feed(m, _cumulative(24, prompt=11))
    assert um["input_tokens"] == 11          # 不是 11 × 24
    assert um["output_tokens"] == 23         # 最后一个 chunk 的累计值
    assert um["total_tokens"] == 34


def test_只在末尾报一次用量的规范通道不受影响():
    # 规范通道只在最后一个 chunk 带 usage,增量等于它本身,结果不该被改动
    m = get_chat_model()
    um = _feed(m, [{"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}])
    assert um["input_tokens"] == 100 and um["output_tokens"] == 20


def test_嵌套的details也按增量算():
    m = get_chat_model()
    usages = [{"prompt_tokens": 50, "completion_tokens": i, "total_tokens": 50 + i,
               "completion_tokens_details": {"reasoning_tokens": i}} for i in range(5)]
    um = _feed(m, usages)
    assert um["input_tokens"] == 50
    assert um["output_token_details"]["reasoning"] == 4


def test_同一实例跑第二条流时基准会自愈():
    # 结构化输出的链会拿同一个模型多次 invoke。不清基准的话,第二条流的增量会算成负数被压成 0
    m = get_chat_model()
    first = _feed(m, _cumulative(5, prompt=11))
    second = _feed(m, _cumulative(5, prompt=11))
    assert second == first
