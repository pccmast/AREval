# AREval — Agent Regression Evaluation Harness

> **Evaluating AI agents at scale. Preventing regressions before they reach production.**

AREval is an open-source evaluation and regression harness for AI agents. It provides a complete toolkit for benchmarking agent performance, detecting quality regressions, and maintaining production-grade reliability across agent iterations.

## Features

- **4 Judge Modes** — Exact-match, LLM-as-a-Judge, Agent-as-a-Judge (with real search + code execution), and DAG-based multi-criteria evaluation
- **13 Metrics** — Accuracy, semantic similarity, RAG triad (faithfulness / relevance / context-precision), tool-call accuracy, safety red-team metrics
- **Statistical Regression Detection** — Paired t-test (scipy) + Cohen's d effect size, severity classification
- **Offline-First** — Every judge and metric degrades gracefully when no API key is available; zero-config `pytest`-friendly CI mode
- **Pluggable Storage** — JSON files (default), SQLite via SQLAlchemy; switch with one environment variable
- **Dashboard** — Next.js 15 + TypeScript real-time visualisation
- **Trace Correlation** — OpenTelemetry-compatible span model with OTLP export

---

## Quick Start

```bash
# 1. Install (includes scipy for statistical tests)
pip install -e ".[all]"

# 2. Run your first evaluation — no API key needed, uses offline mock mode
areval run --dataset datasets/seed/customer_service.jsonl

# 3. View results
cat .areval/runs/*.json

# 4. Launch the dashboard
areval dashboard
```

### Demo Script

The project root includes `demo.py` — a **17-chapter comprehensive feature walkthrough** covering all public APIs:

- **Ch 01–06**: 13 evaluation metrics (exact-match / semantic similarity / RAG triad / agent behavior / safety red-team)
- **Ch 07**: 3 judge modes (LLMJudge / AgentJudge / DAGJudge)
- **Ch 08**: Evaluator pipeline (chain API + full workflow)
- **Ch 09**: Statistical regression detection (paired t-test + Cohen's d + baseline management)
- **Ch 10–17**: Dataset management / storage backends / distributed tracing / online evaluation / SDK decorators / CI/CD reporters / serialization / summary

```bash
# Zero-config run (offline mock mode, ~15 seconds)
python demo.py
```

All features auto-degrade to heuristic offline mode when no API key is available — ideal for quick project overview, onboarding, and interview prep.

---

## Configuration

### API Keys (optional — offline mode works without them)

| Variable | Used by | Required for |
|----------|---------|-------------|
| `OPENAI_API_KEY` | LLMJudge, SemanticSimilarityMetric, RAG metrics | Real LLM evaluation (gpt-4o-mini default) |
| `ANTHROPIC_API_KEY` | LLMJudge (provider="anthropic") | Claude-based evaluation |

Set them as environment variables:

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

When no key is present every component auto-degrades:
- LLM judges → heuristic Jaccard-based mock (score range 0.25–0.95)
- Semantic similarity → pseudo-random deterministic vectors
- **The full evaluation pipeline runs without any API key.**

### Storage Backend

```bash
# Default — JSON files under .areval/runs/
areval run --dataset my_cases.jsonl

# SQLite — indexed, persistent, supports concurrent readers
AREVAL_STORE=sqlite areval run --dataset my_cases.jsonl
```

The storage layer is abstracted behind an `EvaluationStore` ABC. Adding PostgreSQL or Redis in the future means implementing 5 methods — no business code changes.

### CORS (API server)

```bash
# Allow multiple origins (comma-separated)
AREVAL_CORS_ORIGINS="http://localhost:3000,https://my-dashboard.example.com"
```

### Other Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AREVAL_STORE` | `json` | Storage backend: `json` or `sqlite` |
| `AREVAL_DB_PATH` | `.areval/evaluations.db` | SQLite database path |
| `AREVAL_RUNS_DIR` | `.areval/runs` | JSON store directory |
| `AREVAL_CORS_ORIGINS` | `http://localhost:3000` | CORS allowed origins |

---

## CLI Commands

```bash
# Run an evaluation with a config file
areval run --config eval_config.yaml --dataset test_cases.jsonl

# Create a baseline from an evaluation run
areval baseline create --run-id <run_id> --name "v1-production"

# Compare current results against a baseline
areval compare --current results.json --baseline <baseline_id>

# Auto-curate test cases from trace data
areval curate --traces .areval/traces

# Launch the API server
uv run python -m areval_api.main
# or
uv run uvicorn areval_api.main:app --host 0.0.0.0 --port 8700

# Launch the dashboard
areval dashboard --port 3000
```

### Dashboard

A Next.js 15 real-time evaluation dashboard with pass rate trends, run history,
dataset management, and regression tracking.

**Startup sequence**:

```bash
# Terminal 1: Start the API server (required for live data)
uv run python -m areval_api.main

# Terminal 2: Launch the dashboard
areval dashboard --port 3000
```

> **No API? No problem.** The dashboard has built-in mock data fallback.
> Without the API running it shows placeholder data and a banner:
> "Mock data — start the API server to see live data". No white screen.

**Or run just the API in Docker**:

```bash
docker compose up api
# API → http://localhost:8700
```

> Run the dashboard locally (`areval dashboard --port 3000`) to avoid
> Docker image pull issues from restricted networks.

Open `http://localhost:3000` — the header shows connection status:
- `Connected to API — 42 evaluations` → live data from the API
- `Mock data — start the API server to see live data` → API not running, showing placeholders

---

Start the server then hit these endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/datasets` | List datasets |
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

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (135 passing)
pytest tests/ -v

# Run a specific module
pytest tests/test_metrics.py -v
pytest tests/test_judges.py -v
```

```bash
# Run example scripts (standalone templates — copy and adapt)
uv run python examples/basic_evaluation.py
uv run python examples/agent_with_tools.py
uv run python examples/online_monitoring.py
uv run python examples/red_team_evaluation.py

# Or run the full-feature demo (17 chapters, covers entire API)
uv run python demo.py
```

---

## Architecture

```
AREval
├── areval-engine        # Core evaluation engine (Python)
│   ├── metrics/         # 13 built-in metrics
│   ├── judges/          # 4 judge modes (LLM / Agent / DAG / base)
│   ├── regression/      # Statistical regression detection
│   ├── datasets/        # Dataset management + auto-curation
│   ├── storage/         # Pluggable storage (JSON / SQLite)
│   ├── online/          # Real-time evaluation engine
│   ├── tracing/         # OTEL-compatible spans + OTLP export
│   └── evaluator.py     # Evaluation orchestrator
├── areval-sdk           # Python SDK (@eval_trace decorator, CI Reporter)
├── areval-api           # FastAPI REST service (15 endpoints)
├── areval-cli           # Typer + Rich CLI (6 sub-commands)
├── areval-dashboard     # Next.js 15 + TypeScript dashboard
├── tests/               # 135 pytest tests
├── datasets/            # Seed datasets (customer service, RAG, safety)
└── examples/            # Usage examples
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Engine | Python 3.10+, Pydantic 2, NumPy, SciPy |
| API | FastAPI + Uvicorn |
| CLI | Typer + Rich |
| Dashboard | Next.js 15 + TypeScript + Tailwind + Recharts |
| Storage | JSON files / SQLite (SQLAlchemy ORM) |
| Testing | pytest (135 tests), mypy, basedpyright |

## License

MIT
