"""第一个例子：在代码里跟大模型说一句话。"""

from shop import MODEL, client

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "到手不喜欢能退吗？"}],
)
print(resp.choices[0].message.content)
print(resp.usage)
