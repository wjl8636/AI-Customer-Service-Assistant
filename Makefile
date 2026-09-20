# Python 强制 UTF-8。中文 Windows 默认 GBK,读本仓库里带中文的 markdown / jsonl
# (知识库、评估集)会抛 UnicodeDecodeError。必须在 Python 启动前生效,所以设在这里而不是 .env。
export PYTHONUTF8 = 1

.PHONY: help kb-repatch dev test eval eval-check judge-check seed eval-ch07 eval-agent kb-preview kb-build kb-vectorize kb-mine eval-retrieval seed-conv eval-mining kb-reset milvus-up milvus-down smoke-rag eval-rag smoke-interrupt eval-ch06 mcp-up mcp-down langfuse-up langfuse-down

# 目标一多就记不住谁要什么前置条件,带 ## 注释的挑出来列一遍
help:  ## 列出带说明的目标
	@grep -E '^[a-z][a-z0-9-]*:.*?## ' $(MAKEFILE_LIST) \
	  | sed 's/:.*## /\t/' | sort | awk -F'\t' '{printf "  %-22s %s\n", $$1, $$2}'

# ch09: Langfuse 自部署观测栈(web:3000 + worker + postgres + clickhouse + redis + minio)
# -p 独立 project 名:不加会与主 compose 同落 mewhelp project,两边的 minio 服务同名相撞
langfuse-up:
	docker compose -p mewhelp-langfuse -f docker-compose.langfuse.yml up -d
	@echo "Langfuse 起中: http://localhost:3000 (admin@mewhelp.local / mewhelp123)"
	@echo "首次就绪约 2-3 分钟;key 已 headless 预置,写 .env:"
	@echo "  LANGFUSE_PUBLIC_KEY=pk-lf-mewhelp-local"
	@echo "  LANGFUSE_SECRET_KEY=sk-lf-mewhelp-local"
	@echo "  LANGFUSE_BASE_URL=http://localhost:3000"

langfuse-down:
	docker compose -p mewhelp-langfuse -f docker-compose.langfuse.yml down

calibrate-confidence:  ## ch09 置信度阈值校准(需 milvus + 上游可调通 + 知识库已建)
	PYTHONPATH=. uv run python scripts/calibrate_confidence.py

flywheel:  ## ch09 飞轮批处理:问题池 → 标准化查重 → 待审队列(需 mysql + 聊天上游)
	PYTHONPATH=. uv run python scripts/flywheel_pipeline.py

flywheel-samples:  ## ch09 标准化查重 prompt 标注样例验证(通过率 ≥ 80%)
	PYTHONPATH=. uv run python scripts/validate_flywheel_samples.py

eval-flywheel:  ## ch09 评估流水线:复用 ch04 评估集,落 eval_runs 连趋势(TRIGGER=手动|定时)
	PYTHONPATH=. uv run python scripts/eval_flywheel.py --triggered-by $(or $(TRIGGER),手动)

cost-report:  ## ch09 按意图 token 账(需 Langfuse 在跑;DAYS=窗口天数)
	PYTHONPATH=. uv run python scripts/cost_by_intent.py --days $(or $(DAYS),7)

# ch10: 主题分类器(数据→训练→评测→ONNX→旁路批量归类)
ch10-golden:  ## ch10 预标 prompt 黄金样例验证(通过率 ≥ 80% 才放行批量预标;需聊天上游)
	PYTHONPATH=. uv run python scripts/ch10/validate_golden.py

ch10-corpus:  ## ch10 语料流水线:捞池→清洗→预标→模拟补足→导出人工抽审(需 mysql + 聊天上游)
	PYTHONPATH=. uv run python scripts/ch10/build_corpus.py

ch10-dataset:  ## ch10 分层划分 80/10/10 + 训练集增强(需聊天上游)
	PYTHONPATH=. uv run python scripts/ch10/build_dataset.py

ch10-train:  ## ch10 RoBERTa-wwm-ext 全参微调(MPS/CUDA/CPU 自适应,重依赖走 ml 组)
	PYTHONPATH=. uv run --group ml python scripts/ch10/train.py

ch10-eval:  ## ch10 测试集评测:每类 P/R/F1 + 混淆矩阵 + 容错红线 + 判错样本导出
	PYTHONPATH=. uv run --group ml python scripts/ch10/evaluate.py

ch10-export:  ## ch10 导出 ONNX 并校验与 torch 预测完全一致
	PYTHONPATH=. uv run --group ml python scripts/ch10/export_onnx.py

ch10-threshold-scan:  ## ch10 阈值扫描重演:九候选线各算一遍 micro-F1(需 :8110)
	PYTHONPATH=. uv run python scripts/ch10/scan_threshold_replay.py

classifier-up:  ## ch10 推理服务 :8110(ONNX 轻运行时)
	@mkdir -p log data
	@PYTHONPATH=. nohup uv run --group ml python scripts/ch10/serve.py > log/classifier.log 2>&1 & echo $$! > data/classifier.pid
	@sleep 2 && curl -sf http://127.0.0.1:8110/healthz >/dev/null && echo "分类器服务已拉起: :8110(pid 见 data/classifier.pid)" || echo "启动失败,看 log/classifier.log"

classifier-down:
	-@kill `cat data/classifier.pid 2>/dev/null` 2>/dev/null; rm -f data/classifier.pid
	@echo "分类器服务已停"

classify-pool:  ## ch10 旁路批量归类:攒够一批归一次,写 topic_classifications(需 :8110;FORCE=1 不足一批也跑)
	PYTHONPATH=. uv run python scripts/ch10/classify_pool.py $(if $(FORCE),--force,)

dev:
	bash ./scripts/dev.sh

# 后台跑 dev 时的关停(前台跑直接 Ctrl+C)。dev.sh 不写 pid,应用按启动命令找
dev-down:
	-@$(MAKE) mcp-down
	-@pkill -f "uvicorn app.main:app" 2>/dev/null; echo "应用已停"

# ch08: 两台业务 MCP Server 起停(独立进程,Streamable HTTP :8101/:8102)
mcp-up:
	@mkdir -p log data
	@nohup uv run python mcp_servers/logistics_server.py  > log/mcp-logistics.log 2>&1 & echo $$! > data/mcp-logistics.pid
	@nohup uv run python mcp_servers/aftersales_server.py > log/mcp-aftersales.log 2>&1 & echo $$! > data/mcp-aftersales.pid
	@sleep 1 && echo "MCP Servers 已拉起: logistics=:8101 aftersales=:8102(pid 见 data/*.pid)"

mcp-down:
	-@kill `cat data/mcp-logistics.pid 2>/dev/null` 2>/dev/null; rm -f data/mcp-logistics.pid
	-@kill `cat data/mcp-aftersales.pid 2>/dev/null` 2>/dev/null; rm -f data/mcp-aftersales.pid
	@echo "MCP Servers 已停"

# ch04: Milvus Standalone 起停 + RAG 冒烟 + 四策略评估
milvus-up:
	docker compose up -d etcd minio milvus-standalone
	@echo "等待 Milvus 就绪(healthz)..."; \
	for i in $$(seq 1 60); do \
	  curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && echo "Milvus OK" && exit 0; \
	  sleep 3; done; echo "Milvus 未就绪" && exit 1

milvus-down:
	docker compose stop etcd minio milvus-standalone

smoke-rag:
	PYTHONPATH=. uv run python scripts/smoke_milvus_bm25.py
	PYTHONPATH=. uv run python scripts/smoke_rerank.py

eval-check:  ## ch04 评估集自检(ground truth 可命中 + 要点逐字可查;需 Milvus)
	PYTHONPATH=. uv run python scripts/validate_eval_ch04.py

eval-rag:  ## 四策略评估(SKIP_GEN=1 只跑确定性两段,裁判上游挂了时用)
	PYTHONPATH=. uv run python scripts/eval_ch04.py $(if $(SKIP_GEN),--skip-gen,)

# 拿人工处置过的编造个案台账回头考裁判:改完裁判提示词跑这个,不必重跑整轮评估
judge-check:  ## 忠实度裁判回归(重放台账个案的证据+答案,和人工处置对齐率)
	PYTHONPATH=. uv run python scripts/judge_check.py

seed:
	docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch02-seed.sql

test:
	uv run pytest -v

eval:
	uv run python scripts/eval_extract.py

eval-agent:
	uv run python scripts/eval_agent.py

eval-ch05:
	uv run python scripts/eval_ch05.py

smoke-interrupt:  ## ch06 interrupt/resume/astream 中断 surface 红线冒烟
	uv run python scripts/smoke_interrupt.py

eval-ch06:  ## ch06 四验收端到端评估(需全服务起 + 已应用 sql/ch06-ticket-type.sql)
	uv run python scripts/eval_ch06.py

kb-preview:
	PYTHONPATH=. uv run python scripts/show_kb.py

kb-build:
	PYTHONPATH=. uv run python scripts/build_kb.py

kb-vectorize:
	PYTHONPATH=. uv run python scripts/vectorize_kb.py

# 改完 data/kb/*.md 只重嵌改动的那几块(整库重建会清掉飞轮写回的块,补库犯不上)
kb-repatch:
	PYTHONPATH=. uv run python scripts/kb_repatch.py

kb-mine:
	PYTHONPATH=. uv run python scripts/mine_knowledge.py

eval-retrieval:
	PYTHONPATH=. uv run python scripts/eval_retrieval.py

seed-conv:
	docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch03-seed.sql

eval-mining:
	PYTHONPATH=. uv run python scripts/eval_mining.py

# 干净重建知识库:清 MySQL 两表 + drop Milvus collection(ch04 Standalone 无独占锁,app 可不停)
kb-reset:
	docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SET FOREIGN_KEY_CHECKS=0; DELETE FROM knowledge_chunks; DELETE FROM qa_extraction_staging; SET FOREIGN_KEY_CHECKS=1;"
	PYTHONPATH=. uv run python -c "from app.kb import milvus_client as m; m.drop(m.get_client(), 'knowledge')"
	@echo "KB 已重置(Milvus Standalone drop collection)。重跑: make kb-build && make kb-vectorize"

eval-ch07:
	uv run python -m scripts.eval_ch07

eval-ch08:  ## ch08 验收样例端到端(需 make dev 全服务 + MCP Server 在跑 + 已应用 sql/ch08-ddl.sql)
	uv run python -m scripts.eval_ch08
