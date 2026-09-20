# MewHelp 部署任务书

这份文件是写给 AI 编程助手执行的。把它交给 Claude Code、Cursor 之类的工具，说一句
「按 DEPLOY.md 把项目部署起来」就行，它会自己走完下面的步骤。

想自己动手、或者想知道每一步在干什么，看 `README.md`。

---

## 目标

在本机把 MewHelp 跑起来，最终验收标准只有一条：

**浏览器打开 <http://localhost:8000>，问「订单 1001 的物流到哪了」，能看到回答，
并且回答上方出现两个工具调用标记（`query_order` 和 `query_logistics`）。**

只出文字、不出工具调用标记，不算完成。

## 前置条件（执行前先确认，缺了就告诉用户，不要自己乱装）

| 项 | 检查命令 | 缺了怎么办 |
|---|---|---|
| uv | `uv --version` | macOS `brew install uv`；Linux `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | `docker compose version` | 要用户自己装 Docker Desktop 或 Colima 并启动 |
| make | `make --version` | macOS/Linux 一般自带 |
| 三个模型密钥 | 见第 2 步 | **必须问用户要，不要编造，也不要从别处翻** |

内存 8G 起，磁盘 10G 起。Python 不用管，uv 会自己下。

---

## 第 1 步 装依赖

```bash
uv sync
```

会跑几分钟，中间长时间没输出是正常的，不要中断重试。

**验收**：命令退出码为 0，`.venv/` 目录已生成。

## 第 2 步 配置 .env

```bash
cp .env.example .env
```

然后编辑 `.env`，填这三项：

```bash
CHAT_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-v4-flash
CHAT_API_KEY=<用户提供的聊天密钥>
EMBED_API_KEY=<用户提供的硅基流动密钥>
RERANK_API_KEY=<同一个硅基流动密钥>
```

嵌入和重排走硅基流动（`bge-m3` 和 `bge-reranker-v2-m3` 在那边免费），这两位填同一个值；
聊天上游用户自己选，上面给的是 DeepSeek 官方。地址那两行有代码默认值，不用填。

**密钥必须问用户要。** 不要从环境变量、其他项目、历史记录里翻，也不要写假值糊弄过去 ——
假值会让第 4 步静默产出错误结果，比直接失败更难查。

**验收**：`.env` 存在，三个 `*_API_KEY` 都非空。不要把密钥内容打印到终端。

## 第 3 步 起依赖容器

```bash
docker compose up -d
```

拉起 MySQL、Milvus、MinIO、etcd 四个容器。建表不用管，`sql/` 已经挂进 MySQL 的
初始化目录，首次启动会自动按序执行。

**验收**：`docker compose ps` 里 mysql 和 milvus 都是 healthy。

> **头半分钟 MinIO 显示 starting 是正常的**，Docker 每 30 秒才做一次健康检查，刚起来那会儿
> 第一次还没轮到。等一下它会变成 healthy，四个容器最终都是 healthy。
> 要是它一直停在 unhealthy，那是真出问题了，别当成正常现象放过去。

> **镜像拉不下来**（国内常见）：Docker Hub 和 quay.io 经常连不上。先确认镜像站活着 ——
> `curl -sI https://docker.m.daocloud.io/v2/` **返回 401 是正常的**，那是 registry 的
> 鉴权挑战，不是失败。然后走镜像站拉、再打回原名：
>
> ```bash
> docker pull docker.m.daocloud.io/library/mysql:8
> docker tag  docker.m.daocloud.io/library/mysql:8  mysql:8
> docker pull docker.m.daocloud.io/minio/minio:RELEASE.2024-05-28T17-19-04Z
> docker tag  docker.m.daocloud.io/minio/minio:RELEASE.2024-05-28T17-19-04Z minio/minio:RELEASE.2024-05-28T17-19-04Z
> docker pull quay.m.daocloud.io/coreos/etcd:v3.5.16
> docker tag  quay.m.daocloud.io/coreos/etcd:v3.5.16  quay.io/coreos/etcd:v3.5.16
> docker pull docker.m.daocloud.io/milvusdb/milvus:v2.6.22
> docker tag  docker.m.daocloud.io/milvusdb/milvus:v2.6.22  milvusdb/milvus:v2.6.22
> ```
>
> 打回原名之后 `docker-compose.yml` 一个字都不用改。mysql 镜像有 1.1GB，
> **给足时间，不要用短超时去 pull**，没拉完就被中断会让人误以为镜像站不通。

> **3306 被占**：本机如果已经有别的 MySQL 容器占着 3306，`docker compose up` 会失败。
> 告诉用户端口冲突，让用户决定是停掉那个容器还是改端口，**不要自作主张停用户的容器**。

## 第 4 步 灌测试数据

```bash
make seed        # FAQ(query_faq 的数据源)
make seed-conv   # 历史会话(ch03 挖知识用)
```

订单和物流不在这里，它们是 `app/tools/business.py` 里按 user_id 稳定生成的模拟数据，
不落库，任何账号名下都有 1001 和 2002 两笔演示单。数据库里没有 orders 表是正常的。

第 3 步首启时这两个脚本已经自动跑过一遍了，这里再跑是为了重复部署或数据被改乱时重灌。
种子脚本幂等，重复执行不会出问题。

**验收**：`docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SELECT COUNT(*) FROM faq;"`
返回的数字大于 0。

## 第 5 步 建知识库

```bash
make kb-build       # 切 chunk 落 MySQL
make kb-vectorize   # 向量化写进 Milvus，这步会调用上游嵌入接口
```

**验收**：`kb-vectorize` 输出里 Milvus 现有条数大于 0。

这一步失败通常是密钥或网络问题，不是代码问题 —— 先回到第 2 步核对密钥，不要改代码。

## 第 6 步 启动

```bash
make dev
```

这条命令会拉起依赖容器、等 Milvus 健康检查通过、拉起两台 MCP Server（:8101 / :8102），
最后启动应用（:8000）。

**它是前台常驻进程**，需要放到后台执行，否则会一直阻塞。停的时候用 `make dev-down`
（关掉 MCP Server 和应用进程）。人工前台跑的话，Ctrl+C 就停了。启动要等几十秒，
因为它会等 Milvus 就绪才继续，这是正常的。

**验收**：`curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/` 返回 200。

## 第 7 步 验收

```bash
printf '{"user_id":"u1","message":"订单 1001 的物流到哪了"}' > /tmp/q.json
curl -s --max-time 90 http://localhost:8000/api/agent \
     -H 'Content-Type: application/json' --data-binary @/tmp/q.json
```

**验收**：返回的 JSON 里 `tool_calls` 数组包含 `query_order` 和 `query_logistics` 两项，
`answer` 里有物流状态。

只有 `answer` 没有 `tool_calls`，说明模型没有触发工具调用，通常是 `CHAT_MODEL` 写的
名字上游不认，或者中转端点不校验模型名、拿了别的模型应答。回到第 2 步核对。

---

## 完成之后

用一段话告诉用户：跑通了、访问地址是 <http://localhost:8000>、以及验收那一步实际
调用了哪几个工具。

不要顺手改动项目代码。这份任务书里的步骤如果和实际不符，报告差异，让用户来定。
