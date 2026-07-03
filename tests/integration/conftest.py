"""Shared fixtures for integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import pytest
from fastapi.testclient import TestClient

from areval.datasets import DatasetManager
from areval.regression.baseline import BaselineManager
from areval.test_case import TestCase
from areval.storage import JsonFileStore

# ---------------------------------------------------------------------------
# Seed test cases
# ---------------------------------------------------------------------------


def _make_test_cases() -> List[TestCase]:
    """Return a small deterministic set of test cases for API testing."""
    return [
        TestCase(
            id="case-001",
            name="simple-math",
            input="What is 2 + 2?",
            expected_output="4",
            tags=["math", "approved"],
        ),
        TestCase(
            id="case-002",
            name="capital",
            input="What is the capital of France?",
            expected_output="Paris",
            tags=["geography", "approved"],
        ),
        TestCase(
            id="case-003",
            name="pending-review",
            input="Explain quantum computing.",
            expected_output="",
            tags=["science", "pending_review"],
        ),
        TestCase(
            id="case-004",
            name="pending-review-2",
            input="What is machine learning?",
            expected_output="",
            tags=["ai", "pending_review"],
        ),
    ]


# ---------------------------------------------------------------------------
# API client fixture (module-scoped to avoid re-import overhead)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client() -> Generator[TestClient, None, None]:
    """Yield a FastAPI TestClient backed by temporary storage.

    Each test module gets a fresh, isolated DatasetManager / storage
    so that tests don't interfere with real dev data.
    """
    tmp = tempfile.mkdtemp(prefix="areval-test-")
    tmp_path = Path(tmp)
    datasets_dir = tmp_path / ".areval" / "datasets"
    runs_dir = tmp_path / ".areval" / "runs"
    datasets_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)

    # Seed a dataset into the temp store
    mgr = DatasetManager(storage_path=datasets_dir)
    seed = _make_test_cases()
    mgr.create_from_list(
        test_cases=seed, name="API Test Dataset", description="Seed data for integration tests"
    )

    # Storage for evaluation runs
    store = JsonFileStore(directory=runs_dir)

    # ------------------------------------------------------------------
    # Patch module-level singletons in areval_api.main BEFORE test clients
    # hit them.  The app is imported lazily inside the fixture body so
    # that patching happens first.
    # ------------------------------------------------------------------
    import areval_api.main as api_module  # noqa: E402

    api_module.dataset_manager = mgr
    api_module._store = store
    api_module.baseline_manager = BaselineManager(storage_path=tmp_path / ".areval" / "baselines")  # type: ignore[assignment]

    client = TestClient(api_module.app)
    yield client


# ---------------------------------------------------------------------------
# Convenience helpers for tests
# ---------------------------------------------------------------------------


def create_evaluation(
    client: TestClient,
    dataset_id: str,
    agent_type: str = "mock",
    metrics: List[str] | None = None,
    use_judge: bool = False,
    max_cases: int = 3,
) -> Dict[str, Any]:
    """Call POST /api/v1/evaluations and return response body (expects 200)."""
    resp = client.post(
        "/api/v1/evaluations",
        json={
            "dataset_id": dataset_id,
            "agent_type": agent_type,
            "metrics": metrics or ["exact_match"],
            "use_judge": use_judge,
            "max_cases": max_cases,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()
