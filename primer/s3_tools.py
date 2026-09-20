"""第三个例子：给它一份工具清单，它只会说要调哪个，真去查的是你的代码。"""

import json

from shop import MODEL, client, query_order

TOOLS = [{
    "type": "function",
    "function": {
        "name": "query_order",
        "description": "根据订单号查询订单，返回商品、状态和快递单号",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号，形如 SO20260901"}},
            "required": ["order_id"],
        },
    },
}]

messages = [{"role": "user", "content": "我的订单 SO20260901 发货了吗？"}]

resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
msg = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("tool_calls:", msg.tool_calls)

messages.append(msg.model_dump(exclude_none=True))
for call in msg.tool_calls:
    result = query_order(**json.loads(call.function.arguments))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})

resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
print("客服:", resp.choices[0].message.content)
