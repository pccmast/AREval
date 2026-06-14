# AREval — Agent 回归评估平台

> **大规模评估 AI Agent，在上线前拦截质量退化。**

AREval 是一个面向 AI Agent 的开源评估与回归测试平台。提供完整的 Agent 性能基准测试、质量回归检测和生产级可靠性保障。

AREval 与 Agent Gateway（请求路由）和 LLM Scheduling（执行调度）共同构成 **Agent 基础设施栈**中的**质量守门员**。

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 基础设施栈                           │
├───────────┬───────────────┬─────────────────────────────────┤
│  Agent    │  LLM          │  AREval (质量保障层)             │
│  Gateway  │  Scheduling   │  · 评估引擎                      │
│  (路由)   │  (调度)       │  · 回归检测                      │
│           │               │  · 链路追踪                      │
│           │               │  · Dashboard & 报告              │
└───────────┴───────────────┴─────────────────────────────────┘
```

## 特性

- **多模式评估** — 4 种模式：确定性检查、语义相似度、LLM-as-a-Judge、Agent-as-a-Judge
- **回归检测** — 自动基线对比、统计显著性检验（t-test + Cohen's d）、CI/CD 准入控制
- **链路关联** — 与 Gateway 和 Scheduler 的 trace_id 深度关联，端到端可观测
- **插件架构** — Metrics、Judges 通过注册机制可插拔扩展
- **Dashboard** — Next.js 可视化面板，实时评估结果、趋势分析、回归告警
- **数据集管理** — 版本化测试集，JSONL/CSV/SWE-bench 多格式支持

## 快速开始

```bash
# 安装
pip install -e ".[dev,openai]"

# 运行评估（无需 API key，自动降级到 offline/mock 模式）
areval run --dataset test_cases.jsonl --output results.json

# 创建基线
areval baseline create --results results.json

# 对比回归
areval compare --current results.json --baseline <id>

# 启动 Dashboard
areval dashboard
```

## 项目架构

```
AREval
├── areval-engine        # 核心评估引擎（Python）
│   ├── metrics/         # 9 种内置指标（精确匹配、语义相似度、RAG 等）
│   ├── judges/          # 3 种评判模式（LLM/Agent/DAG）
│   ├── regression/      # 统计回归检测（paired t-test + Cohen's d）
│   ├── datasets/        # 数据集管理（JSONL/CSV/SWE-bench）
│   ├── tracing/         # OpenTelemetry 兼容链路追踪
│   └── evaluator.py     # 评估编排引擎
├── areval-sdk           # Python SDK（@eval_trace 装饰器、CI Reporter）
├── areval-api           # FastAPI REST 服务
├── areval-cli           # Typer + Rich 命令行工具
├── areval-dashboard     # Next.js 15 + TypeScript 可视化面板
├── tests/               # 60 个 pytest 测试用例
└── examples/            # 使用示例
```

## 技术栈

| 层 | 技术 |
|----|------|
| 核心引擎 | Python 3.10+, Pydantic, NumPy |
| API | FastAPI + Uvicorn |
| CLI | Typer + Rich（彩色终端界面） |
| Dashboard | Next.js 15 + TypeScript + Tailwind + Recharts |
| 测试 | pytest (60 用例), mypy, basedpyright |
| 存储 | JSON 文件（可升级至 PostgreSQL） |

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行示例
python examples/basic_evaluation.py
python examples/agent_with_tools.py
```

## 许可证

MIT
