"""第四个例子：把工具调用包进一个循环，它自己决定下一步干什么。"""

import json

from shop import MODEL, client, query_logistics, query_order

FUNCS = {"query_order": query_order, "query_logistics": query_logistics}
TOOLS = [
    {"type": "function", "function": {
        "name": "query_order",
        "description": "根据订单号查询订单，返回商品、状态和快递单号",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    }},
    {"type": "function", "function": {
        "name": "query_logistics",
        "description": "根据快递单号查询物流进度和预计送达时间",
        "parameters": {"type": "object", "properties": {"tracking_no": {"type": "string"}}, "required": ["tracking_no"]},
    }},
]
MAX_STEPS = 8


def run_agent(question):
    messages = [{"role": "user", "content": question}]
    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            name, args = call.function.name, json.loads(call.function.arguments)
            try:
                result = FUNCS[name](**args)
            except Exception as exc:
                result = {"error": str(exc)}
            print(f"[第 {step} 步] {name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    return "查询步骤太多了，帮您转人工客服"


print("客服:", run_agent("我的订单 SO20260901 那双鞋到哪了？几号能到？"))
