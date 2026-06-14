"""Tests for BaselineManager regression comparison.

Covers :class:`~areval.regression.baseline.BaselineManager` CRUD
and comparison operations.
"""

import tempfile
from pathlib import Path

import pytest

from areval.regression.baseline import BaselineManager
from areval.test_case import (
    TestCase,
    AgentOutput,
    TestResult,
    EvaluationRun,
    TestStatus,
)


@pytest.fixture
def isolated_mgr():
    """Return a BaselineManager that uses a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield BaselineManager(storage_path=Path(tmpdir))


class TestBaselineManager:
    """CRUD and comparison tests for BaselineManager."""

    def test_create_and_get_baseline(self, isolated_mgr):
        mgr = isolated_mgr
        tc = TestCase(id="b1", name="baseline1", input="x", expected_output="y")
        ao = AgentOutput(output="y")
        tr = TestResult(
            test_case=tc, agent_output=ao,
            status=TestStatus.PASSED, overall_score=0.95, threshold=0.7,
        )
        run = EvaluationRun(name="run-for-baseline")
        run.test_results = [tr]
        run._compute_aggregates()

        bl = mgr.create_baseline(run, name="my-baseline", tags=["prod"])
        assert bl.id.startswith("baseline-")
        assert bl.name == "my-baseline"
        assert bl.tags == ["prod"]
        assert len(bl.test_results) == 1

        fetched = mgr.get_baseline(bl.id)
        assert fetched is not None
        assert fetched.name == "my-baseline"

    def test_get_latest_baseline(self, isolated_mgr):
        mgr = isolated_mgr
        run1 = EvaluationRun(name="old")
        mgr.create_baseline(run1, name="old", tags=["v1"])

        run2 = EvaluationRun(name="new")
        mgr.create_baseline(run2, name="new", tags=["v2"])

        latest = mgr.get_latest_baseline()
        assert latest is not None
        assert latest.name == "new"

        latest_v1 = mgr.get_latest_baseline(tag="v1")
        assert latest_v1 is not None
        assert latest_v1.name == "old"

    def test_list_baselines(self, isolated_mgr):
        mgr = isolated_mgr
        run1 = EvaluationRun(name="r1")
        mgr.create_baseline(run1, name="first")
        run2 = EvaluationRun(name="r2")
        mgr.create_baseline(run2, name="second")
        baselines = mgr.list_baselines()
        assert len(baselines) == 2
        # most recent first
        assert baselines[0].name == "second"

    def test_delete_baseline(self, isolated_mgr):
        mgr = isolated_mgr
        run = EvaluationRun(name="to-delete")
        bl = mgr.create_baseline(run, name="delete-me")
        assert mgr.get_baseline(bl.id) is not None
        assert mgr.delete_baseline(bl.id) is True
        assert mgr.get_baseline(bl.id) is None
        assert mgr.delete_baseline(bl.id) is False

    def test_no_baseline_returns_none(self, isolated_mgr):
        mgr = isolated_mgr
        assert mgr.get_baseline("nonexistent") is None
        assert mgr.get_latest_baseline() is None

    def test_compare_to_baseline(self, isolated_mgr):
        mgr = isolated_mgr
        tc = TestCase(id="c1", name="compare-test", input="q", expected_output="a")
        ao = AgentOutput(output="a")

        # Baseline: score 0.90
        tr_base = TestResult(
            test_case=tc, agent_output=ao,
            status=TestStatus.PASSED, overall_score=0.90, threshold=0.7,
        )
        run_base = EvaluationRun(name="base")
        run_base.test_results = [tr_base]
        run_base._compute_aggregates()
        mgr.create_baseline(run_base, name="golden")

        # Current: score 0.60 (degraded)
        tr_curr = TestResult(
            test_case=tc, agent_output=ao,
            status=TestStatus.FAILED, overall_score=0.60, threshold=0.7,
        )
        comparison = mgr.compare_to_baseline([tr_curr], delta_threshold=0.05)
        comp = comparison["comparisons"][0]
        assert comp["baseline_score"] == 0.90
        assert comp["current_score"] == 0.60
        assert comp["delta"] == pytest.approx(-0.30)
        assert comp["regressed"] is True
        assert comparison["avg_delta"] == pytest.approx(-0.30)
        assert comparison["regressed_count"] == 1

    def test_compare_to_baseline_no_regression(self, isolated_mgr):
        mgr = isolated_mgr
        tc = TestCase(id="c2", name="stable", input="q", expected_output="a")
        ao = AgentOutput(output="a")

        tr_base = TestResult(
            test_case=tc, agent_output=ao,
            status=TestStatus.PASSED, overall_score=0.90, threshold=0.7,
        )
        run_base = EvaluationRun(name="base2")
        run_base.test_results = [tr_base]
        run_base._compute_aggregates()
        mgr.create_baseline(run_base, name="golden2")

        tr_curr = TestResult(
            test_case=tc, agent_output=ao,
            status=TestStatus.PASSED, overall_score=0.89, threshold=0.7,
        )
        comparison = mgr.compare_to_baseline([tr_curr], delta_threshold=0.05)
        assert comparison["comparisons"][0]["delta"] == pytest.approx(-0.01)
        assert comparison["comparisons"][0]["regressed"] is False
