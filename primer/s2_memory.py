"""第二个例子：它记不住，得你自己把聊过的话再发一遍。"""

from shop import MODEL, client


def ask(messages):
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content, resp.usage.prompt_tokens


print("== 两次各问各的 ==")
print(ask([{"role": "user", "content": "我叫喵喵"}])[0])
print(ask([{"role": "user", "content": "我叫什么？"}])[0])

print("\n== 把聊过的话一起发过去 ==")
history = []
for text in ["我叫喵喵", "我叫什么？"]:
    history.append({"role": "user", "content": text})
    reply, tokens = ask(history)
    history.append({"role": "assistant", "content": reply})
    print(f"你: {text}\n客服: {reply}\n[这一轮发出去 {tokens} 个 token]\n")
