"""第五个例子：同一个问题换成 Workflow，先查什么后查什么由代码写死，模型只负责把结果说成人话。"""

import json
import re

from shop import MODEL, client, query_logistics, query_order


def run_workflow(question):
    order_id = re.search(r"SO\d+", question).group()
    order = query_order(order_id)
    if "error" in order:
        return "没查到这个订单，帮您转人工客服"

    logistics = query_logistics(order["tracking_no"])
    facts = json.dumps({"订单": order, "物流": logistics}, ensure_ascii=False)
    resp = client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": "你是电商客服。只根据给你的数据回答，简短口语，数据里没有的不许编。"},
        {"role": "user", "content": f"用户问：{question}\n数据：{facts}"},
    ])
    return resp.choices[0].message.content


print("客服:", run_workflow("我的订单 SO20260901 那双鞋到哪了？几号能到？"))
