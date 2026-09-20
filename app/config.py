# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 必填,故意不给默认值。模型名是跟着账号走的,给一个默认值等于替学员猜:猜错不会在
    # 启动时报错,而是在第一次调模型时抛一个看不懂的上游 401。没有默认值,缺了就在启动
    # 时报 chat_model Field required,直接指向 .env。调优常量(top_k、步数、阈值这些)
    # 是另一类,它们该留在代码里带默认值。
    chat_model: str
    # 三组上游各自直连,不经网关。base_url 和 key 跟着账号走,所以和 chat_model 一样必填;
    # 嵌入和重排都在硅基流动、地址固定,给了默认值,只有 key 要填。
    chat_base_url: str
    chat_api_key: str
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str
    # 思考链控制。三项都留空就用上游默认,填了才传。
    # 关不关是上游的能力、不是参数写法:同一个 thinking.type,硅基流动的 DeepSeek 和
    # MiniMax-M3 上能关(实测 22.4s/318 token → 1.4s/16),MiniMax-M2.x 上会被静默忽略。
    chat_thinking: str = ""           # disabled 关 / adaptive 开
    chat_reasoning_effort: str = ""   # 开着时的强度:low / high / max
    chat_reasoning_split: str = ""    # MiniMax 专属:true 让思考进 reasoning_content 独立字段,
                                      # 而不是混在 content 里的 <think>。关不掉思考链时靠它兜底
    token_budget: int = 2000
    database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp_ch02_test"
    # ch03 知识库检索
    embed_model: str = "BAAI/bge-m3"          # 上游真实名(无网关别名可用)
    milvus_uri: str = "http://localhost:19530"   # ch04: Lite → Standalone
    # ch04 混合检索 + 重排
    rerank_model: str = "BAAI/bge-reranker-v2-m3"   # 上游真实名
    recall_top_k: int = 50        # dense/BM25 各召回 Top-50
    rerank_top_k: int = 10        # 精排出 Top-10
    subquery_split: bool = True   # 一问多意图时按子句拆开各检索一遍再轮转合并(见 retrieval)
    rerank_min_score: float = 0.3 # 重排最高分低于此 → 检索证据低,拒答
    # ch05 图编排
    max_agent_steps: int = 6              # ReAct 环最大步数(封顶,超则兜底)
    # token 花销不在环里卡:那是成本控制,归 ch09 跟 Langfuse 的账一起看。
    # tokens_used 仍逐步累加,进 trace 供排障和统计,只是不再当停止条件。
    checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"  # LangGraph checkpointer(data/*.db* 已 gitignore)
    # ch06 意图识别模型配置(先求准:默认大模型;降级路仅种子,本章不实现运行时)
    # 意图识别、摘要各是一个模型槽位:三项留空就跟 CHAT 用同一家,
    # 想把某个用途换到别家上游(比如摘要走更便宜的托管方)才填这两组。
    intent_base_url: str = ""         # 空=回落 chat_base_url
    intent_api_key: str = ""          # 空=回落 chat_api_key
    summary_base_url: str = ""        # 空=回落 chat_base_url
    summary_api_key: str = ""         # 空=回落 chat_api_key
    intent_model: str = ""            # 意图识别模型名;空=回落 chat_model
    intent_small_model: str = ""      # 降级路小模型(种子,未接运行时)
    intent_mode: str = "accuracy"     # accuracy=只用大模型;cost=小模型判→低置信升级大模型(未实现)
    intent_conf_threshold: float = 0.6  # cost 模式升级阈值(种子)
    # ch07 会话上下文管理(滑窗 + 异步摘要)
    context_window_max_tokens: int = 0    # 滑窗 token 上限。0=按 core/budget 的预算算,
                                          # 填非 0 则手动覆盖(回滚用)
    summary_model: str = ""               # 摘要模型,空=回落 chat_model
    # ch07+ 上下文预算:窗口按块分配,系统提示/检索证据/输出各占一块,滑窗拿剩下的
    model_context_window: int = 0           # 模型窗口。留 0 表示按 chat_model 名字查表(见 core/budget),
                                            # 填非 0 则以这里为准。OpenAI 兼容接口拿不到这个值,只能本地配
    context_budget_turns: int = 20          # 预算按多少轮的最坏情况留;这个量以内的对话不会被裁掉
    max_user_input_tokens: int = 2000       # X:单轮用户输入上限,约 1400 汉字
    max_output_tokens: int = 10000          # Y:单轮输出上限,兼作 API max_tokens。实测最长答复
                                            # 872 字≈1000 token,窗口 1M 下留 1w 只占 1%
    system_prompt_tokens: int = 700         # 实测:AGENT_SYSTEM 208 + 5 个内置工具 schema 476
    evidence_tokens_per_chunk: int = 160    # 实测:answer 99 字 + 问法 10 + 标题路径 20,加编号格式
    # 层 2(半压层)的形态:用户原话一字不动,只压客服答复和大块工具结果
    layer1_ratio: float = 0.7               # 层 1(原文)占滑窗预算的比例,其余归层 2
    layer2_reply_keep_chars: int = 60       # 降到层 2 的客服答复留多少字,首句通常就是结论
    layer2_tool_keep_tokens: int = 200      # 工具结果小于这个数就原样留,大块的压成一行
    tool_result_max_tokens: int = 3000    # 单条工具结果回灌模型的硬上限,执行引擎按它截断。
                                          # 不卡就是无界的:实测 query_faq 单次返回 4251 token
    agent_step_ai_tokens: int = 500       # ReAct 中间步 AI 消息的估值:tool_call 参数,
                                          # 外加关不掉思考链时混在 content 里的那段
    summary_inject_segments: int = 3        # 注入时带最近几段摘要
    summary_tokens_per_segment: int = 250   # 每段摘要的估算体积,几十到两百字
    budget_safety_margin: int = 1000        # 估算误差余量
    # count_tokens_approximately 默认 chars_per_token=4.0 按英文定,中文偏低(实测 29 个汉字
    # 估成 12 token)。中文语料按 1 字≈1.2 token 校准。
    zh_chars_per_token: float = 1.2
    # ch08 工具系统(注册中心 + 执行引擎 + MCP 接入)
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"    # 物流 MCP Server
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"   # 售后 MCP Server
    tool_default_timeout: float = 5.0    # 内置工具默认超时(秒)
    mcp_tool_timeout: float = 10.0       # MCP 工具默认超时(走 HTTP,放宽)
    tool_max_retries: int = 2            # 只读工具暂时性故障最大重试次数
    demo_ticket_delay_seconds: float = 0.0  # 验收 6:>0 时 create_ticket 人为变慢(写超时演示)
    # ch09 可观测(Langfuse 自部署;三者齐全才挂回调,缺省时系统照常跑、测试环境不依赖)
    # 环境变量名与课程 README/SDK 一致:LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = ""     # 如 http://localhost:3000(自部署地址,数据不出门)
    # ch09 置信度闸(正式版):阈值由 make calibrate-confidence 在 ch04 评估集上校准回填,不拍脑袋
    # 实测:可答桶 p50=0.704 vs 应拒桶 p50=0.102/max=0.290,YoudenJ 最优点 0.17
    # 选 0.30 而非 0.17:误放代价远大于误拦,宁可多转人工也不放编造答案
    # (可答通过率 82.9%、应拒放行率 0%,报告见 data/ch09/reports/confidence_calibration.txt,
    #  「观测与成本」页 /observability 上也能看,推荐值与这里不一致时页面会标出来)
    evidence_confidence_threshold: float = 0.30


settings = Settings()
