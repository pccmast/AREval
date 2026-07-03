# AREval — Agent Regression Evaluation Harness

[English](README.md) | [中文](README_CN.md)

> **Evaluating AI agents at scale. Preventing regressions before they reach production.**

AREval is an open-source evaluation and regression harness for AI agents. It provides a complete toolkit for benchmarking agent performance, detecting quality regressions, and maintaining production-grade reliability across agent iterations.

## Quick Start

```bash
# 1. Install everything (engine + API + dashboard deps)
make install

# 2. Start dev environment — API + Dashboard with hot reload
make dev

# 3. Or run a quick evaluation from CLI
areval run --dataset examples/test_cases.jsonl
```

> **No API key? No problem.** All judges and metrics auto-degrade to heuristic offline mode. The full pipeline runs with zero config.

---

## Development vs Production

|  | Dev (`make dev`) | Production (Docker) |
|---|---|---|
| **API** | `uvicorn --reload :8700` (hot reload) | `docker compose up api` |
| **Dashboard** | `npm run dev :3000` (HMR) | Static export / CDN |
| **Storage** | Local `.areval/` | Persistent volume |
| **When to use** | Writing code, debugging | Deploying, demos |

```bash
make dev          # One command, two services, hot reload on both
make docker-run   # Production-style deployment
```

---

## Makefile Commands

All common tasks have shorthand commands:

| Command | What it does |
|---|---|
| `make install` | Install Python + Node dependencies |
| `make dev` | Start API (:8700) + Dashboard (:3000) with hot reload |
| `make test` | Run 262 unit tests |
| `make test-integration` | Run 26 API integration tests |
| `make check` | Full CI pipeline: test + integration + lint + format |
| `make lint` | Ruff + mypy |
| `make format` | Black auto-format |
| `make docker-build` | Build Docker images |
| `make docker-run` | Start with Docker Compose |
| `make clean` | Remove build artifacts |

---

## Features

- **4 Judge Modes** — Exact-match, LLM-as-a-Judge, Agent-as-a-Judge (with tool access), DAG multi-criteria evaluation
- **13 Metrics** — Accuracy, semantic similarity, RAG triad, tool-call accuracy, safety red-team metrics
- **Statistical Regression Detection** — Paired t-test (scipy) + Cohen's d effect size, severity classification
- **Offline-First** — Every judge and metric degrades gracefully when no API key is available
- **Pluggable Storage** — JSON files (default), SQLite; switch with one environment variable
- **Dashboard** — Next.js 15 + TypeScript real-time visualisation
- **CI/CD Ready** — 288 tests (262 unit + 26 integration), mypy type checking, ruff linting, Docker smoke tests
- **Docker Optimized** — Layer-cached builds, `.dockerignore`, GHA cache in CI

---

## Configuration

### API Keys (optional)

| Variable | Used by | Required for |
|---|---|---|
| `OPENAI_API_KEY` | LLMJudge, SemanticSimilarity, RAG metrics | Real LLM evaluation |
| `ANTHROPIC_API_KEY` | LLMJudge (provider="anthropic") | Claude-based evaluation |

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

When no key is present:
- LLM judges → heuristic Jaccard mock (score 0.25–0.95)
- Semantic similarity → deterministic vectors
- The full pipeline runs without any API key.

### Storage

```bash
# Default — JSON files
areval run --dataset my_cases.jsonl

# SQLite
AREVAL_STORE=sqlite areval run --dataset my_cases.jsonl
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AREVAL_STORE` | `json` | Storage backend: `json` or `sqlite` |
| `AREVAL_DB_PATH` | `.areval/evaluations.db` | SQLite database path |
| `AREVAL_RUNS_DIR` | `.areval/runs` | JSON store directory |
| `AREVAL_CORS_ORIGINS` | `http://localhost:3000` | CORS allowed origins |

---

## API Endpoints

Start the server (`make api` or `make dev`) and access:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/datasets` | List datasets |
| `GET`/`PUT`/`POST` | `/api/v1/datasets/{id}/*` | Dataset CRUD + review workflow |
| `POST` | `/api/v1/evaluations` | Run a new evaluation |
| `GET` | `/api/v1/evaluations` | List evaluation runs |
| `GET` | `/api/v1/evaluations/{id}` | Get run details |
| `GET` | `/api/v1/evaluations/{id}/results` | Get per-test results |
| `GET` | `/api/v1/baselines` | List baselines |
| `GET` | `/api/v1/metrics` | List available metrics |
| `GET` | `/api/v1/stats` | Aggregate statistics |
| `POST` | `/api/v1/online/evaluate` | Real-time single evaluation |
| `GET` | `/api/v1/online/stats` | Online eval stats (time window) |
| `GET` | `/api/v1/online/trend` | Time-series trend data |
| `GET` | `/api/v1/online/health` | Online quality health |
| `GET` | `/api/v1/online/alerts` | Alert history |

---

## Architecture

```
AREval
├── areval-engine        # Core evaluation engine
│   ├── metrics/         # 13 built-in metrics
│   ├── judges/          # 4 judge modes
│   ├── regression/      # Statistical regression detection
│   ├── datasets/        # Dataset management + auto-curation
│   ├── storage/         # Pluggable storage (JSON / SQLite)
│   ├── online/          # Real-time evaluation engine
│   └── tracing/         # OTEL-compatible spans
├── areval-sdk           # Python SDK (@eval_trace decorator, CI Reporter)
├── areval-api           # FastAPI REST service (17 endpoints)
├── areval-cli           # Typer + Rich CLI
├── areval-dashboard     # Next.js 15 + TypeScript dashboard
├── tests/               # 288 pytest tests (262 unit + 26 integration)
└── examples/            # Usage examples + demo scripts
```

## Testing

```bash
make test              # 262 unit tests
make test-integration  # 26 API integration tests
make check             # Full CI: tests + lint + format
```

```bash
# Run examples
uv run python examples/basic_evaluation.py
uv run python examples/agent_with_tools.py

# Full 17-chapter feature walkthrough
uv run python demo.py
```

## Tech Stack

| Layer | Technology |
|---|---|
| Engine | Python 3.10+, Pydantic 2, NumPy, SciPy |
| API | FastAPI + Uvicorn |
| CLI | Typer + Rich |
| Dashboard | Next.js 15 + TypeScript + Tailwind + Recharts |
| Storage | JSON / SQLite (SQLAlchemy ORM) |
| CI | GitHub Actions (unit + integration + mypy + ruff + Docker smoke) |

## License

MIT
