"""FastAPI REST API for AREval.

Provides HTTP endpoints for:
- Running evaluations
- Managing datasets and baselines
- Querying results and traces
- WebSocket for real-time updates
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(
    title="AREval API",
    description="Agent Regression Evaluation Harness API",
    version="0.1.0",
)

# CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Managers
dataset_manager = DatasetManager()
baseline_manager = BaselineManager()

# In-memory storage for demo (use DB in production)
_eval_runs: Dict[str, EvaluationRun] = {}


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
async def create_evaluation(
    dataset_id: str,
    metrics: List[str] = Query(["exact_match"]),
    threshold: float = 0.7,
    use_judge: bool = False,
) -> Dict[str, Any]:
    """Start a new evaluation run.

    In production: This would queue a background job.
    For the project: Returns a mock run for demonstration.
    """
    dataset = dataset_manager.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build evaluator
    evaluator = Evaluator(threshold=threshold)

    metric_map = {
        "exact_match": ExactMatchMetric(),
        "contains": ContainsMetric(),
        "semantic_similarity": SemanticSimilarityMetric(),
        "faithfulness": FaithfulnessMetric(),
    }

    for m in metrics:
        if m in metric_map:
            evaluator.add_metric(metric_map[m])

    if use_judge:
        evaluator.add_judge(LLMJudge())

    # Note: In production, agent_fn would be provided
    # For demo, create a simple mock agent
    def mock_agent(tc: TestCase) -> AgentOutput:
        return AgentOutput(
            output=f"Mock response for: {tc.input[:50]}...",
            latency_ms=100.0,
            token_usage={"input": 10, "output": 20},
        )

    eval_run = evaluator.evaluate(
        test_cases=dataset.test_cases[:5],  # Limit for demo
        agent_fn=mock_agent,
        run_name=f"api-evaluation-{dataset.name}",
    )

    _eval_runs[eval_run.id] = eval_run

    return {
        "run_id": eval_run.id,
        "status": "completed",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
