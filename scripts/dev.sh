#!/usr/bin/env bash
# 单入口拉起 FastAPI 应用(:8000);三组上游各自直连,没有网关进程
set -euo pipefail
cd "$(dirname "$0")/.."

# Python 强制 UTF-8。中文 Windows 默认编码是 GBK,读本仓库里带中文的 markdown / jsonl
# (知识库、评估集)会抛 UnicodeDecodeError。这个变量必须在 Python 启动前设好,写进 .env 太晚。
export PYTHONUTF8=1

if [ ! -f .env ]; then
  echo "缺少 .env,请 cp .env.example .env 并填入 token" >&2
  exit 1
fi
set -a; source .env; set +a

# --- preflight:确保依赖基建活着(Colima → 容器 → Milvus 就绪)---
# 没有这步的话,Docker VM 一挂,应用只打个「Milvus 预热异常,继续启动」的非致命
# warning 就带故障起来了,知识类问答全哑,很难排查。
if command -v colima >/dev/null 2>&1; then
  if ! colima status >/dev/null 2>&1; then
    echo "Colima 未运行,启动中(30-60s)..."
    colima start
  fi
fi

echo "拉起依赖容器(mysql/etcd/minio/milvus)..."
docker compose up -d mysql etcd minio milvus-standalone

echo "等待 Milvus 就绪(healthz)..."
MILVUS_READY=0
for _ in $(seq 1 60); do
  if curl -sf http://localhost:9091/healthz >/dev/null 2>&1; then
    MILVUS_READY=1
    break
  fi
  sleep 3
done
if [ "$MILVUS_READY" -ne 1 ]; then
  echo "Milvus 90s 内未就绪,退出(知识类问答会不可用)" >&2
  exit 1
fi
echo "Milvus OK"

make mcp-up   # ch08: 物流/售后 MCP Server(独立进程 :8101/:8102)

uv run uvicorn app.main:app --port 8000
