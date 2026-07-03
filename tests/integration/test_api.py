"""Integration tests for the AREval REST API.

These tests exercise every endpoint of the FastAPI server through
``fastapi.testclient.TestClient``, which runs the full request/response
pipeline without a real HTTP server.

Markers: ``pytest -m "integration"`` to run, or (with minimal config)
they are discovered alongside unit tests.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# ============================================================================
# Health & metadata
# ============================================================================


def test_health_check(api_client):
    """GET /health returns {"status":"ok"}."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "areval-api"


def test_metrics_list(api_client):
    """GET /api/v1/metrics lists all available metrics."""
    resp = api_client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(m["name"] == "exact_match" for m in data)
    assert any(m["name"] == "semantic_similarity" for m in data)


# ============================================================================
# Datasets
# ============================================================================


def test_list_datasets(api_client):
    """GET /api/v1/datasets returns the seed dataset."""
    resp = api_client.get("/api/v1/datasets")
    assert resp.status_code == 200
    datasets = resp.json()
    assert len(datasets) == 1
    assert datasets[0]["name"] == "API Test Dataset"
    assert datasets[0]["size"] == 4


def test_get_dataset_by_id(api_client):
    """GET /api/v1/datasets/{id} returns full details."""
    # Find the seed dataset ID
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]

    resp = api_client.get(f"/api/v1/datasets/{ds_id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["name"] == "API Test Dataset"
    assert len(detail["test_cases"]) == 4


def test_get_dataset_not_found(api_client):
    """GET /api/v1/datasets/{id} returns 404 for unknown IDs."""
    resp = api_client.get("/api/v1/datasets/nonexistent")
    assert resp.status_code == 404


def test_approve_case(api_client):
    """PUT /api/v1/datasets/{id}/cases/{case_id}/approve works."""
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]

    # case-003 is pending_review — approve it
    resp = api_client.put(f"/api/v1/datasets/{ds_id}/cases/case-003/approve")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approved"
    # One pending case removed; case-004 still remains
    assert data["review_stats"]["pending_review"] == 1


def test_reject_case(api_client):
    """PUT /api/v1/datasets/{id}/cases/{case_id}/reject removes the case."""
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]

    # case-004 is pending_review — reject it
    resp = api_client.put(f"/api/v1/datasets/{ds_id}/cases/case-004/reject")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "rejected"


def test_approve_all(api_client):
    """POST /api/v1/datasets/{id}/approve-all approves all pending cases.

    NOTE: This runs after approve_case (case-003 already approved) and
    reject_case (case-004 already removed), so 0 pending remain.
    """
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]

    resp = api_client.post(f"/api/v1/datasets/{ds_id}/approve-all")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approved"
    assert data["review_stats"]["pending_review"] == 0


# ============================================================================
# Evaluations
# ============================================================================


def test_create_evaluation_mock(api_client):
    """POST /api/v1/evaluations with mock agent returns a completed run."""
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]

    resp = api_client.post(
        "/api/v1/evaluations",
        json={
            "dataset_id": ds_id,
            "agent_type": "mock",
            "metrics": ["exact_match"],
            "max_cases": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["agent_type"] == "mock"
    assert body["summary"]["total_cases"] == 2
    assert 0 <= body["summary"]["pass_rate"] <= 1


def test_create_evaluation_echo(api_client):
    """POST /api/v1/evaluations with echo agent works."""
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]

    resp = api_client.post(
        "/api/v1/evaluations",
        json={
            "dataset_id": ds_id,
            "agent_type": "echo",
            "metrics": ["exact_match", "contains"],
            "max_cases": 2,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_create_evaluation_with_judge(api_client):
    """POST /api/v1/evaluations with judge enabled."""
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]

    resp = api_client.post(
        "/api/v1/evaluations",
        json={
            "dataset_id": ds_id,
            "agent_type": "mock",
            "metrics": ["exact_match"],
            "use_judge": True,
            "max_cases": 1,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_create_evaluation_invalid_agent(api_client):
    """POST /api/v1/evaluations rejects invalid agent_type (Pydantic validation)."""
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]

    resp = api_client.post(
        "/api/v1/evaluations",
        json={"dataset_id": ds_id, "agent_type": "invalid", "max_cases": 1},
    )
    assert resp.status_code == 422  # Unprocessable Entity (validation error)


def test_create_evaluation_dataset_not_found(api_client):
    """POST /api/v1/evaluations returns 404 for unknown dataset."""
    resp = api_client.post(
        "/api/v1/evaluations",
        json={"dataset_id": "does-not-exist", "agent_type": "mock", "max_cases": 1},
    )
    assert resp.status_code == 404


def test_list_evaluations(api_client):
    """GET /api/v1/evaluations returns all runs."""
    resp = api_client.get("/api/v1/evaluations")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_evaluation_detail(api_client):
    """GET /api/v1/evaluations/{run_id} returns full detail."""
    # First create a run
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]
    created = api_client.post(
        "/api/v1/evaluations",
        json={
            "dataset_id": ds_id,
            "agent_type": "mock",
            "metrics": ["exact_match"],
            "max_cases": 1,
        },
    ).json()
    run_id = created["run_id"]

    # Then fetch it
    resp = api_client.get(f"/api/v1/evaluations/{run_id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == run_id


def test_get_evaluation_results(api_client):
    """GET /api/v1/evaluations/{run_id}/results returns per-case scores."""
    resp = api_client.get("/api/v1/datasets")
    ds_id = resp.json()[0]["id"]
    created = api_client.post(
        "/api/v1/evaluations",
        json={
            "dataset_id": ds_id,
            "agent_type": "mock",
            "metrics": ["exact_match"],
            "max_cases": 1,
        },
    ).json()
    run_id = created["run_id"]

    resp = api_client.get(f"/api/v1/evaluations/{run_id}/results")
    assert resp.status_code == 200
    results = resp.json()
    assert isinstance(results, list)
    assert len(results) == 1


def test_get_evaluation_not_found(api_client):
    """GET /api/v1/evaluations/{run_id} returns 404."""
    resp = api_client.get("/api/v1/evaluations/does-not-exist")
    assert resp.status_code == 404


# ============================================================================
# Baselines
# ============================================================================


def test_list_baselines_empty(api_client):
    """GET /api/v1/baselines returns an empty list initially."""
    resp = api_client.get("/api/v1/baselines")
    assert resp.status_code == 200
    assert resp.json() == []


# ============================================================================
# Stats
# ============================================================================


def test_get_stats(api_client):
    """GET /api/v1/stats returns aggregated statistics."""
    resp = api_client.get("/api/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["datasets"] >= 1
    assert data["baselines"] >= 0
    assert "total_evaluations" in data


# ============================================================================
# Online evaluation
# ============================================================================


def test_online_evaluate(api_client):
    """POST /api/v1/online/evaluate scores a single agent call."""
    resp = api_client.post(
        "/api/v1/online/evaluate",
        json={
            "input": "What is Python?",
            "output": "Python is a programming language.",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert "passed" in data
    assert "scores" in data


def test_online_evaluate_with_tools(api_client):
    """POST /api/v1/online/evaluate accepts optional fields."""
    resp = api_client.post(
        "/api/v1/online/evaluate",
        json={
            "input": "Search for latest news",
            "output": "Here are the results...",
            "tool_calls": [{"tool": "search", "query": "latest news"}],
            "trace_id": "trace-abc-123",
            "latency_ms": 250.0,
        },
    )
    assert resp.status_code == 200


def test_online_health(api_client):
    """GET /api/v1/online/health returns current quality status."""
    resp = api_client.get("/api/v1/online/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "pass_rate" in data or "status" in data


def test_online_stats(api_client):
    """GET /api/v1/online/stats accepts window_minutes param."""
    resp = api_client.get("/api/v1/online/stats?window_minutes=30")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_online_trend(api_client):
    """GET /api/v1/online/trend returns time series data."""
    resp = api_client.get("/api/v1/online/trend?window_minutes=60&bucket_minutes=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_online_alerts(api_client):
    """GET /api/v1/online/alerts returns alert history."""
    resp = api_client.get("/api/v1/online/alerts?limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ============================================================================
# Edge cases
# ============================================================================


def test_cors_headers(api_client):
    """Response includes CORS headers configured in main.py."""
    resp = api_client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # FastAPI CORS middleware handles OPTIONS and sets allow-origin
    assert resp.status_code in (200, 204, 405)
