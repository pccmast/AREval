# AREval — Agent Regression Evaluation Harness

> **Evaluating AI agents at scale. Preventing regressions before they reach production.**

AREval is an open-source evaluation and regression harness for AI agents. It provides a complete toolkit for benchmarking agent performance, detecting quality regressions, and maintaining production-grade reliability across agent iterations.

Built to complement [Agent Gateway](https://github.com/yourname/agent-gateway) and [LLM Scheduling](https://github.com/yourname/llm-scheduling), AREval forms the **quality assurance layer** of the agent infrastructure stack.

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Infrastructure Stack                 │
├─────────────┬─────────────┬─────────────────────────────────┤
│ Agent       │   LLM       │   AREval (Quality Assurance)    │
│ Gateway     │ Scheduling  │   • Evaluation Engine           │
│ (Routing)   │ (Execution) │   • Regression Detection        │
│             │             │   • Trace Analysis              │
│             │             │   • Dashboard & Reporting       │
└─────────────┴─────────────┴─────────────────────────────────┘
```

## Features

- **Multi-Mode Evaluation** — LLM-as-a-Judge, Agent-as-a-Judge, programmatic metrics, and human-in-the-loop scoring
- **Regression Harness** — Automated baseline comparison, statistical significance testing, and CI/CD-native gates
- **Trace Correlation** — Deep linking with Gateway and Scheduler traces for end-to-end observability
- **Plugin Architecture** — Extensible metrics, judges, and evaluation pipelines
- **Dashboard** — Real-time evaluation results, trend analysis, and regression alerts
- **Dataset Management** — Versioned test sets, synthetic data generation, and gold-set curation

## Quick Start

```bash
# Install
pip install areval[all]

# Configure
areval configure --api-key $OPENAI_API_KEY

# Run your first evaluation
areval run --config eval_config.yaml --dataset test_cases.jsonl

# Launch dashboard
areval dashboard
```

## Architecture

```
AREval
├── areval-engine      # Core evaluation engine
│   ├── metrics/       # Built-in and custom metrics
│   ├── judges/        # LLM-as-a-Judge, Agent-as-a-Judge
│   ├── runners/       # Evaluation orchestration
│   ├── regression/    # Regression detection algorithms
│   ├── datasets/      # Dataset management
│   ├── tracing/       # OpenTelemetry-compatible tracing
│   └── plugins/       # Plugin system
├── areval-sdk         # Python SDK (decorators, tracers)
├── areval-api         # FastAPI REST service
├── areval-dashboard   # Next.js web dashboard
└── areval-cli         # Command-line interface
```

## License

MIT
