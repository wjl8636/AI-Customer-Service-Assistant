#!/usr/bin/env bash
# 验收:两轮流式对话,第二轮须接住第一轮上下文
set -euo pipefail
SID="demo-$$"

echo "=== 第一轮:自报家门+提商品 ==="
curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d "{\"session_id\": \"$SID\", \"message\": \"我叫王小明,昨天买了你们的智能猫砂盆\"}"
echo; echo "=== 第二轮:考上下文(我叫什么?买了什么?) ==="
curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d "{\"session_id\": \"$SID\", \"message\": \"还记得我叫什么、买了什么吗?\"}"
echo
