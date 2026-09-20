"""上下文信息保留率对照实验 v3。

30 轮客服对话，9 次工具调用，总 ~8000 token。
10 个事实分散在前半段（R2-R11）和中段（R16-R20），
后 10 轮纯填充，不含任何关键信息。

budget = 总量的 50%，模拟系统提示/工具定义/证据/输出各占一半。

baseline: 纯 trim_history 滑窗
optimized: build_window 三层 + 摘要注入，锚点自动算
"""
import asyncio, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+aiosqlite:///")

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.core.memory import (
    trim_history, build_window, count_tokens,
    next_layer1_from, _index_after,
)
from app.core.llm import get_chat_model
from app.core.prompts import SUMMARY_SYSTEM

PLANTED_FACTS = {
    # 前半段（R2-R8）
    "order_id": "MH-20260815-7742",
    "user_name": "林雨萱",
    "product_model": "MH-CAM1",
    "complaint_reason": "镜头有划痕",
    "refund_amount": "299.00元",
    # 中段（R16-R20），baseline 50% 能保住这些
    "payment_method": "花呗分期三期",
    "shipping_address": "杭州市西湖区文三路 478 号",
    "agent_promise": "48小时内安排换货",
    "delivery_date": "8月20号下午3点",
    "coupon_code": "SUMMER2026",
}

RECALL_QUESTIONS = [
    ("用户叫什么名字？", "user_name"),
    ("用户的订单号是什么？", "order_id"),
    ("退款金额是多少？", "refund_amount"),
    ("商品型号是什么？", "product_model"),
    ("商品出了什么质量问题？", "complaint_reason"),
    ("换货寄到哪个地址？", "shipping_address"),
    ("客服承诺多久安排换货？", "agent_promise"),
    ("预计什么时候送达？", "delivery_date"),
    ("用户的优惠券代码是什么？", "coupon_code"),
    ("用户用什么方式付的款？", "payment_method"),
]

_ctr = 0
def _hm(c):
    global _ctr; _ctr += 1
    return HumanMessage(content=c, id=f"db-{_ctr}")
def _am(c, **kw):
    global _ctr; _ctr += 1
    kw["id"] = f"ai-{_ctr}"
    return AIMessage(content=c, **kw)
def _tm(c, cid, name):
    return ToolMessage(content=c, tool_call_id=cid, name=name)


def _big_policy():
    return (
        "【退换货政策 v3.2】\n"
        "一、七天无理由退货：自签收之日起7天内，未拆封可退。\n"
        "二、质量问题：签收30天内，非人为损坏可退换。退款3-5工作日原路退回。\n"
        "三、换货时效：审核通过48小时内发货。\n"
        "四、运费：质量问题商家承担，无理由退货买家承担。\n"
        "五、赠品：主商品退货赠品须一并退回。\n"
    ) * 4

def _big_order():
    F = PLANTED_FACTS
    return (
        f"订单号：{F['order_id']}\n商品：{F['product_model']} 智能摄像头\n"
        f"单价：{F['refund_amount']}\n收货人：{F['user_name']}\n"
        f"下单：2026-08-10 14:32\n签收：2026-08-13 10:22\n状态：已签收\n"
        f"物流：顺丰SF1234567890\n"
    )

def _big_faq(topic):
    return (
        f"【{topic}常见问题】\n"
        "Q1: 设备无法开机？检查供电线，长按重置键5秒。\n"
        "Q2: 画面模糊？擦拭镜头，检查保护膜。质量问题可申请换货。\n"
        "Q3: WiFi连不上？仅支持2.4GHz，检查密码，靠近路由器重试。\n"
        "Q4: 夜视效果差？检查红外灯，避免前方反光物体。有效距离30米。\n"
        "Q5: 录像卡顿？检查TF卡速度等级，建议Class10以上。\n"
        "Q6: App闪退？升级到最新版本，清除缓存重试。\n"
    ) * 3


def make_messages():
    global _ctr; _ctr = 0
    msgs = []
    F = PLANTED_FACTS

    # === 前半段 R1-R11: 埋 5 个事实 + 工具调用 ===

    # R1
    msgs.append(_hm("你好，我买的摄像头有点问题"))
    msgs.append(_am("您好！请告诉我您的订单号，我帮您查。"))

    # R2: user_name, order_id
    msgs.append(_hm(f"我叫{F['user_name']}，订单号{F['order_id']}"))
    msgs.append(_am(f"好的{F['user_name']}，查一下。"))

    # R3: 工具 - 查订单
    tc1 = [{"id": "tc-1", "name": "query_order", "args": {"order_id": F["order_id"]}}]
    msgs.append(_hm("帮我查订单"))
    msgs.append(_am("", tool_calls=tc1))
    msgs.append(_tm(_big_order(), "tc-1", "query_order"))
    msgs.append(_am(f"查到了，{F['product_model']}智能摄像头，已签收。请问什么问题？"))

    # R4: complaint_reason
    msgs.append(_hm(f"收到发现{F['complaint_reason']}"))
    msgs.append(_am(f"{F['complaint_reason']}属于质量问题，我查一下政策。"))

    # R5: 工具 - 查政策
    tc2 = [{"id": "tc-2", "name": "query_faq", "args": {"query": "退换货"}}]
    msgs.append(_hm("能不能退或换"))
    msgs.append(_am("", tool_calls=tc2))
    msgs.append(_tm(_big_policy(), "tc-2", "query_faq"))
    msgs.append(_am("符合质量问题退换条件。退还是换？"))

    # R6: refund_amount
    msgs.append(_hm("退能退多少"))
    msgs.append(_am(f"全额退款{F['refund_amount']}。"))

    # R7: 工具 - 查FAQ
    tc3 = [{"id": "tc-3", "name": "query_faq", "args": {"query": "安装"}}]
    msgs.append(_hm("先问个别的，这个怎么安装"))
    msgs.append(_am("", tool_calls=tc3))
    msgs.append(_tm(_big_faq("安装"), "tc-3", "query_faq"))
    msgs.append(_am("包装里有底座螺丝，固定好接电用App扫码绑定。"))

    # R8-R11: 纯对话填充（无事实，产生距离）
    mid_fillers_1 = [
        ("WiFi有什么要求", "只支持2.4GHz，5G连不上。路由器背面能看到频段信息。"),
        ("夜视效果怎么样", "红外夜视30米，晚上看得很清楚。前面别放反光物体。"),
        ("能装室外吗", "IP67防水防尘，室外完全没问题，工作温度零下20到60度。"),
        ("录像能存多久", "TF卡最大256G存一个月，云存储月费9.9起。两种可以同时用。"),
    ]
    for q, a in mid_fillers_1:
        msgs.append(_hm(q))
        msgs.append(_am(a))

    # R12-R15: 工具调用填充（制造大块工具结果，把前半段挤远）
    fill_tools = [
        ("智能音箱能联动吗", "query_faq", "联动",
         _big_faq("智能联动") + "\n天猫精灵/小爱同学均支持，HomeKit暂不支持。\n" * 3),
        ("App有什么功能", "query_faq", "App功能",
         _big_faq("App功能") + "\n实时监控/录像回放/移动侦测/设备分享均支持。\n" * 3),
        ("数据安全怎么保证", "query_faq", "数据安全",
         _big_faq("数据安全") + "\nTLS1.3传输加密，AES-256存储加密，等保二级认证。\n" * 3),
        ("有没有企业采购方案", "query_faq", "企业采购",
         _big_faq("企业采购") + "\n10台9折，50台85折，200台8折。专属客户经理。\n" * 3),
    ]
    for q, tool, query, result in fill_tools:
        tc = [{"id": f"tc-f-{_ctr}", "name": tool, "args": {"query": query}}]
        msgs.append(_hm(q))
        msgs.append(_am("", tool_calls=tc))
        msgs.append(_tm(result, f"tc-f-{_ctr}", tool))
        msgs.append(_am("相关资料已经查到了，您看看还有别的问题吗？"))

    # === 中段 R16-R20: 埋后 5 个事实 ===
    # 这些在 baseline 50% budget 时应该还在窗口里

    # R16: payment_method
    msgs.append(_hm(f"回到退换货的事，我是{F['payment_method']}付的，能退到花呗吗"))
    msgs.append(_am(f"{F['payment_method']}支付的退款原路返回花呗账户。"))

    # R17: shipping_address
    msgs.append(_hm(f"我决定换货，寄到{F['shipping_address']}"))
    msgs.append(_am(f"换货地址记下了：{F['shipping_address']}。"))

    # R18: agent_promise, delivery_date
    msgs.append(_hm("什么时候能收到"))
    msgs.append(_am(f"我们承诺{F['agent_promise']}，预计{F['delivery_date']}送达。"))

    # R19: coupon_code
    msgs.append(_hm(f"我有优惠券{F['coupon_code']}换货后还能用吗"))
    msgs.append(_am(f"优惠券{F['coupon_code']}换货后仍有效，到9月底。"))

    # R20: 追问确认
    msgs.append(_hm("退回去的旧机器你们怎么处理"))
    msgs.append(_am("质检确认问题后走报废流程，个人数据会先恢复出厂清除。"))

    # === R21-R30: 纯对话填充 ===
    tail_fillers = [
        ("质检要多久", "一到两个工作日，通过立刻触发换货发货。"),
        ("新机器是全新的吗", "全新未拆封，出库前再做一次全检。"),
        ("有延保服务吗", "一年基础保修，可选延保一年或两年，签收30天内购买。"),
        ("延保多少钱", "一年原价5%，两年原价8%。延保期间维修免费。"),
        ("能多个手机同时看吗", "一台设备最多绑5个账号，同时在线看建议不超3个。"),
        ("有移动侦测吗", "有，移动和声音侦测两种，灵敏度三档可调。"),
        ("电费大概多少", "待机3瓦录制5瓦，一个月电费不到3块。"),
        ("你们客服几点下班", "在线客服早九晚十全年无休，夜间可留言。"),
        ("好的没什么问题了", "祝您使用愉快！有问题随时联系。"),
        ("最后确认一下之前聊的", "好的，您想确认哪些信息？"),
    ]
    for q, a in tail_fillers:
        msgs.append(_hm(q))
        msgs.append(_am(a))

    return msgs


async def _ask_recall(model, context_msgs, question):
    prompt = (
        "你是客服助手。根据下面的对话记录回答问题。"
        "记录中没有的信息回答「不知道」。只答问题本身。\n\n"
    )
    for m in context_msgs:
        if isinstance(m, HumanMessage):
            prompt += f"用户: {m.content}\n"
        elif isinstance(m, AIMessage) and m.content:
            prompt += f"客服: {m.content}\n"
        elif isinstance(m, ToolMessage):
            prompt += f"[工具]: {m.content[:200]}...\n"
    prompt += f"\n问题: {question}\n回答: "
    resp = await model.ainvoke([HumanMessage(content=prompt)])
    return resp.content.strip()


def _check(answer, key):
    return PLANTED_FACTS[key] in answer


async def main():
    RUNS = 3
    all_msgs = make_messages()
    total_tok = count_tokens(all_msgs)
    budget = total_tok // 2

    print(f"=== Config ===")
    print(f"msgs={len(all_msgs)}  total_tok={total_tok}  budget={budget} (50%)")

    model = get_chat_model(streaming=False)

    # --- Baseline ---
    bl_window = trim_history(all_msgs, max_tokens=budget)
    bl_tok = count_tokens(bl_window)
    bl_text = " ".join(m.content for m in bl_window if hasattr(m, 'content') and m.content)
    bl_in = sum(1 for v in PLANTED_FACTS.values() if v in bl_text)
    print(f"\nBASELINE: {len(bl_window)} msgs, {bl_tok} tok, facts_in_window={bl_in}/10")
    for key, val in PLANTED_FACTS.items():
        print(f"  {'IN' if val in bl_text else '--'} {key}")

    # --- Optimized ---
    layer1_budget = int(budget * 0.7)
    auto_l1 = next_layer1_from(all_msgs, 0, layer1_budget)
    l2_end = _index_after(all_msgs, auto_l1)
    l2_msgs = all_msgs[:l2_end]
    print(f"\nOPTIMIZED: layer1_from=db-{auto_l1}, l2_zone={len(l2_msgs)} msgs")

    sp = f"{SUMMARY_SYSTEM}\n\n"
    for m in l2_msgs:
        if isinstance(m, HumanMessage): sp += f"用户: {m.content}\n"
        elif isinstance(m, AIMessage) and m.content: sp += f"客服: {m.content}\n"
        elif isinstance(m, ToolMessage): sp += f"[工具]: {m.content[:200]}\n"
    print("  generating summary...")
    sr = await model.ainvoke([HumanMessage(content=sp)])
    summary = sr.content.strip()
    print(f"  summary ({len(summary)} chars): {summary[:80]}...")

    opt_window = build_window(all_msgs, summary_upto_msg_id=auto_l1,
                              layer1_from_msg_id=0, max_tokens=budget)
    summary_msg = HumanMessage(content=f"(earlier summary: {summary})")
    opt_full = [summary_msg] + opt_window
    opt_tok = count_tokens(opt_full)
    opt_text = " ".join(m.content for m in opt_full if hasattr(m, 'content') and m.content)
    opt_in = sum(1 for v in PLANTED_FACTS.values() if v in opt_text)
    print(f"  window: {len(opt_full)} msgs, {opt_tok} tok, facts={opt_in}/10")
    for key, val in PLANTED_FACTS.items():
        in_s = val in summary
        in_w = val in opt_text
        print(f"  {'IN' if in_w else '--'} {key}  (summary={'Y' if in_s else 'N'})")

    # --- Run ---
    bl_scores, op_scores = [], []
    for run in range(RUNS):
        bc = oc = 0
        for q, key in RECALL_QUESTIONS:
            try:
                ba = await _ask_recall(model, bl_window, q)
                if _check(ba, key): bc += 1
            except: pass
            await asyncio.sleep(0.5)
            try:
                oa = await _ask_recall(model, opt_full, q)
                if _check(oa, key): oc += 1
            except: pass
            await asyncio.sleep(0.5)
        bl_scores.append(bc)
        op_scores.append(oc)
        print(f"  run {run+1}: baseline={bc}/10  optimized={oc}/10")

    bl_avg = sum(bl_scores) / len(bl_scores) * 10
    op_avg = sum(op_scores) / len(op_scores) * 10
    print(f"\n{'='*50}")
    print(f"Baseline  avg: {bl_avg:.0f}%  {bl_scores}")
    print(f"Optimized avg: {op_avg:.0f}%  {op_scores}")
    print(f"Delta: +{op_avg - bl_avg:.0f} pp")


if __name__ == "__main__":
    asyncio.run(main())
