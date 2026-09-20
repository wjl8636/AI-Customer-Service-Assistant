# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""意图四件套标注评估:九类判对率(ch08 增「人工」)+ confidence 可解析 + 怪问题落其他 + 多轮漂移。需聊天上游可调通。
用法:.venv/bin/python -m scripts.eval_intent"""
import asyncio

from app.core.intent import classify

SAMPLES = [
    ("订单1001的快递到哪了", "物流"), ("我买的东西发货了吗", "物流"),
    ("订单2002现在什么状态", "订单"), ("我上周下的单多少钱来着", "订单"),
    ("这款猫粮多少钱一包", "商品咨询"), ("退货运费一般谁承担", "商品咨询"),
    ("我要退货", "退款退货"), ("这个订单还能申请退款吗", "退款退货"),
    ("我的猫爬架坏了能保修吗", "售后"), ("换新进度到哪了", "售后"),
    ("你们这什么破服务,我要投诉", "投诉"), ("太差了给我个说法", "投诉"),
    ("帮我建个工单", "人工"), ("猫砂盆漏电了,帮我建个工单跟进", "人工"),
    ("你好呀", "闲聊"), ("今天天气不错", "闲聊"),
    ("帮我写一段 Python 代码", "其他"), ("阿斯顿发发", "其他"),
]


async def main():
    passed = bad_json = 0
    for q, expect in SAMPLES:
        r = await classify(q)
        got = r.get("intent")
        conf = r.get("confidence")
        ok = got == expect
        json_ok = got in ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他") \
            and isinstance(conf, float) and 0.0 <= conf <= 1.0
        bad_json += not json_ok
        passed += ok
        print(f"{'✅' if ok else '❌'} {q!r} -> {got}(conf={conf}) 期望={expect}")

    # 多轮漂移:物流→退款→物流,当前句意图应随上下文
    hist = "用户:订单1001到哪了\n客服:已发货,深圳分拨中心\n用户:那我想退了\n客服:好的,帮您看下退款\n"
    r = await classify("那它现在到哪了", hist)
    print(f"多轮漂移『那它现在到哪了』(退款后)-> {r['intent']}(期望 物流)")

    print(f"\n判对 {passed}/{len(SAMPLES)};JSON 越界 {bad_json} 条(glm 非确定性,抖动如实重跑记录)")


if __name__ == "__main__":
    asyncio.run(main())
