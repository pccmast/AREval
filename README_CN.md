# AREval — Agent 回归评估平台

> **大规模评估 AI Agent，在上线前拦截质量退化。**

AREval 是一个面向 AI Agent 的开源评估与回归测试平台。提供完整的 Agent 性能基准测试、质量回归检测和生产级可靠性保障。

## 快速开始

```bash
# 1. 安装全部依赖（引擎 + API + Dashboard）
make install

# 2. 启动开发环境 — API + Dashboard 双服务热更新
make dev

# 3. 或从命令行快速跑一次评估
areval run --dataset examples/test_cases.jsonl
```

> **无需 API Key**：所有评判器和指标在无 Key 时自动降级为启发式 mock 模式，零配置即可跑通完整流水线。

---

## 开发模式 vs 生产模式

|  | 开发 (`make dev`) | 生产 (Docker) |
|---|---|---|
| **API** | `uvicorn --reload :8700`（热更新） | `docker compose up api` |
| **Dashboard** | `npm run dev :3000`（HMR） | 静态导出 / CDN |
| **存储** | 本地 `.areval/` | 持久化卷 |
| **适用场景** | 写代码、调试 | 部署、演示 |

```bash
make dev          # 一条命令启动双服务，改代码自动刷新
make docker-run   # 生产级容器部署
```

---

## Makefile 快捷命令

所有常用操作都有缩写命令：

| 命令 | 功能 |
|---|---|
| `make install` | 安装 Python + Node 依赖 |
| `make dev` | 一键启动 API (:8700) + Dashboard (:3000)，热更新 |
| `make test` | 运行 262 个单元测试 |
| `make test-integration` | 运行 26 个 API 集成测试 |
| `make check` | 完整 CI 流水线：测试 + 集成测试 + Lint + 格式化 |
| `make lint` | Ruff 代码检查 + mypy 类型检查 |
| `make format` | Black 自动格式化 |
| `make docker-build` | 构建 Docker 镜像 |
| `make docker-run` | Docker Compose 启动 |
| `make clean` | 清理构建产物 |

---

## 核心特性

- **4 种评判模式** — 精确匹配、LLM-as-a-Judge、Agent-as-a-Judge（含工具调用）、DAG 多维度评估
- **13 种内置指标** — 精度、语义相似度、RAG 三件套、工具调用准确性、安全红队指标
- **统计回归检测** — 配对 t 检验（scipy）+ Cohen's d 效应量 + 严重度分级
- **离线优先** — 无 API Key 时全部降级为启发式 mock，零配置 CI 兼容
- **可插拔存储** — JSON 文件（默认）、SQLite；一个环境变量切换
- **Dashboard** — Next.js 15 + TypeScript 实时可视化面板
- **CI/CD 完备** — 288 个测试（262 单元 + 26 集成），mypy 类型检查，ruff 代码检查，Docker 冒烟测试
- **Docker 优化** — 分层缓存构建、`.dockerignore` 安全过滤、GHA 缓存加速

---

## 配置说明

### API Key（可选）

| 环境变量 | 使用者 | 启用能力 |
|---|---|---|
| `OPENAI_API_KEY` | LLMJudge、语义相似度、RAG 指标 | 真实 LLM 评估 |
| `ANTHROPIC_API_KEY` | LLMJudge（provider="anthropic"） | Claude 评估 |

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

无 Key 时自动降级：
- LLM 评判器 → 启发式 Jaccard mock（0.25–0.95 可区分好/坏回答）
- 语义相似度 → 确定性伪随机向量
- 完整流水线无需任何 Key 即可运行

### 存储后端

```bash
# 默认 — JSON 文件
areval run --dataset my_cases.jsonl

# SQLite — 支持索引查询
AREVAL_STORE=sqlite areval run --dataset my_cases.jsonl
```

### 环境变量一览

| 变量 | 默认值 | 说明 |
|---|---|---|
| `AREVAL_STORE` | `json` | 存储后端：`json` 或 `sqlite` |
| `AREVAL_DB_PATH` | `.areval/evaluations.db` | SQLite 数据库路径 |
| `AREVAL_RUNS_DIR` | `.areval/runs` | JSON 存储目录 |
| `AREVAL_CORS_ORIGINS` | `http://localhost:3000` | CORS 允许来源 |

---

## API 端点

启动服务后（`make api` 或 `make dev`）可用：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/datasets` | 列出数据集 |
| `GET`/`PUT`/`POST` | `/api/v1/datasets/{id}/*` | 数据集增删改 + 审核工作流 |
| `POST` | `/api/v1/evaluations` | 创建评估运行 |
| `GET` | `/api/v1/evaluations` | 列出评估运行 |
| `GET` | `/api/v1/evaluations/{id}` | 获取运行详情 |
| `GET` | `/api/v1/evaluations/{id}/results` | 获取逐用例测试结果 |
| `GET` | `/api/v1/baselines` | 列出基线 |
| `GET` | `/api/v1/metrics` | 列出可用指标 |
| `GET` | `/api/v1/stats` | 聚合统计 |
| `POST` | `/api/v1/online/evaluate` | 实时单次评估 |
| `GET` | `/api/v1/online/stats` | 在线评估统计（时间窗口） |
| `GET` | `/api/v1/online/trend` | 时序趋势数据 |
| `GET` | `/api/v1/online/health` | 在线质量健康度 |
| `GET` | `/api/v1/online/alerts` | 告警历史 |

---

## 项目架构

```
AREval
├── areval-engine        # 核心评估引擎
│   ├── metrics/         # 13 种内置指标
│   ├── judges/          # 4 种评判模式
│   ├── regression/      # 统计回归检测
│   ├── datasets/        # 数据集管理 + 自动策展
│   ├── storage/         # 可插拔存储（JSON / SQLite）
│   ├── online/          # 实时评估引擎
│   └── tracing/         # OTEL 兼容 span
├── areval-sdk           # Python SDK（@eval_trace 装饰器、CI Reporter）
├── areval-api           # FastAPI REST 服务（17 个端点）
├── areval-cli           # Typer + Rich CLI
├── areval-dashboard     # Next.js 15 + TypeScript 面板
├── tests/               # 288 个 pytest 用例（262 单元 + 26 集成）
└── examples/            # 使用示例 + demo 脚本
```

## 测试

```bash
make test              # 262 个单元测试
make test-integration  # 26 个 API 集成测试
make check             # 完整 CI：测试 + Lint + 格式化
```

```bash
# 运行示例
uv run python examples/basic_evaluation.py
uv run python examples/agent_with_tools.py

# 17 章全功能演示
uv run python demo.py
```

## 技术栈

| 层 | 技术 |
|---|---|
| 核心引擎 | Python 3.10+, Pydantic 2, NumPy, SciPy |
| API | FastAPI + Uvicorn |
| CLI | Typer + Rich |
| Dashboard | Next.js 15 + TypeScript + Tailwind + Recharts |
| 存储 | JSON / SQLite（SQLAlchemy ORM） |
| CI | GitHub Actions（单元 + 集成 + mypy + ruff + Docker 冒烟） |

## 许可证

MIT
