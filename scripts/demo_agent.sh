#!/usr/bin/env bash
# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
# ch02 验收演示:三条验收标准。需 mewhelp-mysql 容器 + make seed + make dev 就绪。
set -euo pipefail
BASE=http://localhost:8000

echo "== 验收1:订单物流(应选中 query_logistics 并按返回结果回答)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"订单 1001 的物流到哪了"}' | python3 -m json.tool

echo ""
echo "== 验收2:退货政策(query_faq 命中)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"退货政策是什么"}' | python3 -m json.tool

echo ""
echo "== 验收3:邮费是多少(语义鸿沟:答案在「运费怎么算」的 answer,但 keyword「邮费」LIKE question 查不到 → 漏召回=预期,留 ch03 向量检索)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"邮费是多少"}' | python3 -m json.tool
