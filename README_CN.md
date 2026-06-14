# AREval — Agent 回归评估平台

> **大规模评估 AI Agent，在上线前拦截质量退化。**

AREval 是一个面向 AI Agent 的开源评估与回归测试平台。提供完整的 Agent 性能基准测试、质量回归检测和生产级可靠性保障。

## 特性

- **4 种评判模式** — 精确匹配、LLM-as-a-Judge、Agent-as-a-Judge（真实搜索 + 代码执行）、DAG 多维度评估
- **13 种内置指标** — 精度、语义相似度、RAG 三件套（忠实度 / 切题度 / 上下文精度）、工具调用准确性、安全红队指标
- **统计回归检测** — 配对 t 检验（scipy）+ Cohen's d 效应量 + 严重度分级
- **离线优先设计** — 所有评判器和指标在无 API key 时自动降级为启发式 mock，支持零配置 CI 模式
- **可插拔存储** — JSON 文件（默认）、SQLite（SQLAlchemy）；一个环境变量切换
- **Dashboard** — Next.js 15 + TypeScript 实时可视化
- **链路追踪** — OpenTelemetry 兼容 span 模型，支持 OTLP 导出

---

## 快速开始

```bash
# 1. 安装（含 scipy 统计检验依赖）
pip install -e ".[all]"

# 2. 运行首次评估 — 无需 API key，自动使用离线 mock 模式
areval run --dataset datasets/seed/customer_service.jsonl

# 3. 查看结果
cat .areval/runs/*.json

# 4. 启动 Dashboard
areval dashboard
```

---

## 配置说明

### API Key（可选 — 无 Key 时离线模式也能跑）

| 环境变量 | 使用者 | 启用能力 |
|----------|--------|---------|
| `OPENAI_API_KEY` | LLMJudge、语义相似度、RAG 指标 | 真实 LLM 评估（默认 gpt-4o-mini） |
| `ANTHROPIC_API_KEY` | LLMJudge（provider="anthropic"） | Claude 评估 |

设置方法：

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

无 Key 时自动降级：
- LLM 评判器 → 启发式 Jaccard mock（分数范围 0.25–0.95，可区分好/坏回答）
- 语义相似度 → 伪随机确定性向量
- **完整评估流水线无需任何 API key 即可运行。**

### 存储后端切换

```bash
# 默认 — JSON 文件，存储在 .areval/runs/
areval run --dataset my_cases.jsonl

# SQLite — 支持索引查询、持久化
AREVAL_STORE=sqlite areval run --dataset my_cases.jsonl
```

存储层基于 `EvaluationStore` 抽象接口设计，未来切换 PostgreSQL / Redis 只需实现 5 个方法，业务代码零改动。

### CORS 配置（API 服务）

```bash
# 允许多个来源（逗号分隔）
AREVAL_CORS_ORIGINS="http://localhost:3000,https://my-dashboard.example.com"
```

### 其他环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AREVAL_STORE` | `json` | 存储后端：`json` 或 `sqlite` |
| `AREVAL_DB_PATH` | `.areval/evaluations.db` | SQLite 数据库路径 |
| `AREVAL_RUNS_DIR` | `.areval/runs` | JSON 存储目录 |
| `AREVAL_CORS_ORIGINS` | `http://localhost:3000` | CORS 允许来源 |

---

## CLI 命令

```bash
# 用 YAML 配置文件运行评估
areval run --config eval_config.yaml --dataset test_cases.jsonl

# 从评估运行创建基线
areval baseline create --run-id <run_id> --name "v1-production"

# 对比当前结果与基线
areval compare --current results.json --baseline <baseline_id>

# 从 Trace 数据自动策展测试用例
areval curate --traces .areval/traces

# 启动 API 服务
python -m areval_api.main
# 或
uvicorn areval_api.main:app --host 0.0.0.0 --port 8000

# 启动 Dashboard
areval dashboard --port 3000
```

---

## API 端点

启动 API 服务后可用：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/datasets` | 列出数据集 |
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

---

## 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试（135 个用例全部通过）
pytest tests/ -v

# 运行指定模块
pytest tests/test_metrics.py -v
pytest tests/test_judges.py -v
```

```bash
# 运行示例脚本
python examples/basic_evaluation.py
python examples/agent_with_tools.py
```

---

## 项目架构

```
AREval
├── areval-engine        # 核心评估引擎（Python）
│   ├── metrics/         # 13 种内置指标
│   ├── judges/          # 4 种评判模式（LLM / Agent / DAG / base）
│   ├── regression/      # 统计回归检测
│   ├── datasets/        # 数据集管理 + 自动策展
│   ├── storage/         # 可插拔存储（JSON / SQLite）
│   ├── online/          # 实时评估引擎
│   ├── tracing/         # OTEL 兼容 span + OTLP 导出
│   └── evaluator.py     # 评估编排器
├── areval-sdk           # Python SDK（@eval_trace 装饰器、CI Reporter）
├── areval-api           # FastAPI REST 服务（15 个端点）
├── areval-cli           # Typer + Rich CLI（6 个子命令）
├── areval-dashboard     # Next.js 15 + TypeScript 面板
├── tests/               # 135 个 pytest 测试用例
├── datasets/            # 种子数据集（客服、RAG、安全）
└── examples/            # 使用示例
```

## 技术栈

| 层 | 技术 |
|----|------|
| 核心引擎 | Python 3.10+, Pydantic 2, NumPy, SciPy |
| API | FastAPI + Uvicorn |
| CLI | Typer + Rich（彩色终端界面） |
| Dashboard | Next.js 15 + TypeScript + Tailwind + Recharts |
| 存储 | JSON 文件 / SQLite（SQLAlchemy ORM） |
| 测试 | pytest（135 个用例），mypy，basedpyright |

## 许可证

MIT
