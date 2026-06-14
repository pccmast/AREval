"""FastAPI REST API for AREval.

Provides HTTP endpoints for:
- Running evaluations
- Managing datasets and baselines
- Querying results and traces
- WebSocket for real-time updates
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from areval.evaluator import Evaluator
from areval.test_case import TestCase, AgentOutput, EvaluationRun
from areval.datasets import DatasetManager
from areval.regression.baseline import BaselineManager
from areval.metrics import (
    ExactMatchMetric,
    ContainsMetric,
    SemanticSimilarityMetric,
    FaithfulnessMetric,
)
from areval.judges import LLMJudge
from areval.online.evaluator import OnlineEvaluator
from areval.online.storage import TimeSeriesStorage
from areval.online.monitors import QualityMonitor, AlertConfig
from areval.test_case import TestCase as TCase, AgentOutput as AOutput
from areval.utils.serialization import reconstruct_run

app = FastAPI(
    title="AREval API",
    description="Agent Regression Evaluation Harness API",
    version="0.1.0",
)

# CORS for dashboard integration
allowed_origins_str = os.environ.get(
    "AREVAL_CORS_ORIGINS", "http://localhost:3000"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins_str.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Managers
dataset_manager = DatasetManager()
baseline_manager = BaselineManager()

# JSON file-backed evaluation run storage
_EVAL_RUNS_DIR = Path(os.environ.get("AREVAL_RUNS_DIR", ".areval/runs"))
_EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
_eval_runs: Dict[str, EvaluationRun] = {}


def _save_run(run: EvaluationRun) -> None:
    """Persist an evaluation run to a JSON file."""
    file_path = _EVAL_RUNS_DIR / f"{run.id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(run.to_dict(), f, indent=2, default=str)


def _load_all_runs() -> None:
    """Load all persisted evaluation runs from disk."""
    if not _EVAL_RUNS_DIR.exists():
        return
    for file_path in _EVAL_RUNS_DIR.glob("*.json"):
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            run = reconstruct_run(data)
            _eval_runs[run.id] = run
        except (json.JSONDecodeError, KeyError):
            continue


# Load existing runs on startup
_load_all_runs()

# Online evaluation instance (singleton for the API process)
_online_storage = TimeSeriesStorage()
_online_evaluator = OnlineEvaluator(
    metrics=[ExactMatchMetric(), SemanticSimilarityMetric()],
    threshold=0.7,
    storage=_online_storage,
    monitor=QualityMonitor(
        storage=_online_storage,
        config=AlertConfig(window_minutes=30, min_samples=3),
    ),
    async_mode=False,  # sync for API demo
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class CreateEvaluationRequest(BaseModel):
    """Request body for creating a new evaluation run."""

    dataset_id: str = Field(..., description="Dataset ID to evaluate against")
    metrics: List[str] = Field(
        default=["exact_match"],
        description="List of metric names to apply",
    )
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Pass/fail threshold")
    use_judge: bool = Field(default=False, description="Enable LLM judge evaluation")
    agent_type: str = Field(
        default="mock",
        description="Agent type: 'mock', 'echo', or 'llm'",
        pattern="^(mock|echo|llm)$",
    )
    max_cases: int = Field(default=5, ge=1, le=100, description="Max test cases to evaluate")


# ---------------------------------------------------------------------------
# Agent implementations
# ---------------------------------------------------------------------------

def _mock_agent(tc: TestCase) -> AgentOutput:
    """Return a deterministic mock response for demo/CI."""
    return AgentOutput(
        output=f"Mock response for: {tc.input[:50]}...",
        latency_ms=100.0,
        token_usage={"input": 10, "output": 20},
    )


def _echo_agent(tc: TestCase) -> AgentOutput:
    """Return the first 100 characters of the test input as output.

    Useful for repeatable demonstrations where the output is deterministic
    and known in advance.
    """
    return AgentOutput(
        output=tc.input[:100],
        latency_ms=50.0,
        token_usage={"input": len(tc.input.split()), "output": 10},
    )


def _llm_agent_factory(model: str = "gpt-4o-mini") -> Any:
    """Factory that returns an agent function backed by a real LLM.

    Requires OPENAI_API_KEY in environment.
    Falls back to echo behaviour if the key is missing.
    """

    def _llm_agent(tc: TestCase) -> AgentOutput:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return AgentOutput(
                output=f"[LLM unavailable — no OPENAI_API_KEY] Question: {tc.input[:80]}",
                latency_ms=0.0,
            )

        try:
            from openai import OpenAI
        except ImportError:
            return AgentOutput(
                output=f"[openai package not installed] Question: {tc.input[:80]}",
                latency_ms=0.0,
            )

        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": tc.input}],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        return AgentOutput(
            output=content,
            latency_ms=response.usage.total_tokens if response.usage else 0,
            token_usage={
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
            },
        )

    return _llm_agent


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "areval-api"}


@app.get("/api/v1/datasets")
async def list_datasets() -> List[Dict[str, Any]]:
    """List all available datasets."""
    datasets = dataset_manager.list_datasets()
    return [
        {
            "id": d.id,
            "name": d.name,
            "description": d.description,
            "size": d.size,
            "tags": d.tags,
            "created_at": d.created_at.isoformat(),
        }
        for d in datasets
    ]


@app.get("/api/v1/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """Get dataset details."""
    dataset = dataset_manager.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset.to_dict()


@app.post("/api/v1/evaluations")
async def create_evaluation(body: CreateEvaluationRequest) -> Dict[str, Any]:
    """Start a new evaluation run.

    Supports three agent types:
    - 'mock': deterministic fake responses (default, no API key needed)
    - 'echo': repeats the input as output (repeatable demo)
    - 'llm': calls OpenAI API to generate real answers (needs OPENAI_API_KEY)

    The number of test cases evaluated is capped at max_cases (default 5).
    """
    dataset = dataset_manager.get_dataset(body.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Select agent function
    if body.agent_type == "echo":
        agent_fn = _echo_agent
    elif body.agent_type == "llm":
        agent_fn = _llm_agent_factory()
    else:
        agent_fn = _mock_agent

    # Build evaluator
    evaluator = Evaluator(threshold=body.threshold)

    metric_map: Dict[str, Any] = {
        "exact_match": ExactMatchMetric(),
        "contains": ContainsMetric(),
        "semantic_similarity": SemanticSimilarityMetric(),
        "faithfulness": FaithfulnessMetric(),
    }

    for m in body.metrics:
        if m in metric_map:
            evaluator.add_metric(metric_map[m])

    if body.use_judge:
        evaluator.add_judge(LLMJudge())

    eval_run = evaluator.evaluate(
        test_cases=dataset.test_cases[: body.max_cases],
        agent_fn=agent_fn,
        run_name=f"api-evaluation-{dataset.name}",
    )

    _eval_runs[eval_run.id] = eval_run
    _save_run(eval_run)

    return {
        "run_id": eval_run.id,
        "status": "completed",
        "agent_type": body.agent_type,
        "summary": {
            "total_cases": eval_run.total_cases,
            "passed_cases": eval_run.passed_cases,
            "pass_rate": eval_run.pass_rate,
            "avg_score": eval_run.avg_score,
        },
    }


@app.get("/api/v1/evaluations")
async def list_evaluations() -> List[Dict[str, Any]]:
    """List all evaluation runs."""
    return [
        {
            "id": r.id,
            "name": r.name,
            "total_cases": r.total_cases,
            "pass_rate": r.pass_rate,
            "avg_score": r.avg_score,
            "regression_count": r.regression_count,
            "started_at": r.started_at.isoformat(),
        }
        for r in sorted(_eval_runs.values(), key=lambda x: x.started_at, reverse=True)
    ]


@app.get("/api/v1/evaluations/{run_id}")
async def get_evaluation(run_id: str) -> Dict[str, Any]:
    """Get evaluation run details."""
    run = _eval_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run.to_dict()


@app.get("/api/v1/evaluations/{run_id}/results")
async def get_evaluation_results(run_id: str) -> List[Dict[str, Any]]:
    """Get detailed test results for a run."""
    run = _eval_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return [r.to_dict() for r in run.test_results]


@app.get("/api/v1/baselines")
async def list_baselines() -> List[Dict[str, Any]]:
    """List all baselines."""
    baselines = baseline_manager.list_baselines()
    return [
        {
            "id": b.id,
            "name": b.name,
            "run_id": b.run_id,
            "created_at": b.created_at.isoformat(),
            "tags": b.tags,
        }
        for b in baselines
    ]


@app.get("/api/v1/metrics")
async def list_available_metrics() -> List[Dict[str, str]]:
    """List available evaluation metrics."""
    return [
        {"name": "exact_match", "description": "Exact string match", "type": "programmatic"},
        {"name": "contains", "description": "Substring containment", "type": "programmatic"},
        {"name": "regex_match", "description": "Regex pattern match", "type": "programmatic"},
        {"name": "semantic_similarity", "description": "Embedding-based similarity", "type": "model_based"},
        {"name": "faithfulness", "description": "Context faithfulness (RAG)", "type": "model_based"},
        {"name": "answer_relevance", "description": "Answer relevance (RAG)", "type": "model_based"},
        {"name": "tool_call_accuracy", "description": "Tool call correctness", "type": "agent"},
        {"name": "task_completion", "description": "Binary task completion", "type": "agent"},
    ]


@app.get("/api/v1/stats")
async def get_stats() -> Dict[str, Any]:
    """Get overall statistics."""
    runs = list(_eval_runs.values())
    return {
        "total_evaluations": len(runs),
        "total_test_cases": sum(r.total_cases for r in runs),
        "average_pass_rate": sum(r.pass_rate for r in runs) / len(runs) if runs else 0,
        "total_regressions": sum(r.regression_count for r in runs),
        "datasets": len(dataset_manager.list_datasets()),
        "baselines": len(baseline_manager.list_baselines()),
    }


# ---------------------------------------------------------------------------
# Online evaluation endpoints
# ---------------------------------------------------------------------------


class OnlineEvalRequest(BaseModel):
    input: str
    output: str = ""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    trace_id: Optional[str] = None
    latency_ms: float = 0.0


@app.post("/api/v1/online/evaluate")
async def online_evaluate(body: OnlineEvalRequest) -> Dict[str, Any]:
    """Evaluate a single Agent call in real time."""
    tc = TCase(name=f"online-{body.input[:20]}", input=body.input)
    ao = AOutput(
        output=body.output,
        tool_calls=body.tool_calls,
        trace_id=body.trace_id,
        latency_ms=body.latency_ms,
    )
    result = _online_evaluator.evaluate(tc, ao)
    if result is None:
        return {"status": "queued"}
    return {
        "score": result.overall_score,
        "passed": result.passed,
        "scores": result.scores,
    }


@app.get("/api/v1/online/stats")
async def online_stats(window_minutes: int = 60) -> Dict[str, Any]:
    """Get online evaluation statistics for a time window."""
    return _online_evaluator.get_stats(window_minutes=window_minutes)


@app.get("/api/v1/online/health")
async def online_health() -> Dict[str, Any]:
    """Get current health status."""
    return _online_evaluator.get_health()


@app.get("/api/v1/online/trend")
async def online_trend(
    window_minutes: int = 1440,
    bucket_minutes: int = 60,
) -> List[Dict[str, Any]]:
    """Get time-series trend data."""
    return _online_evaluator.get_trend(window_minutes=window_minutes, bucket_minutes=bucket_minutes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
