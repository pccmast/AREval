"""Tests for shared serialization utilities.

Covers :func:`~areval.utils.serialization.reconstruct_test_result`
and :func:`~areval.utils.serialization.reconstruct_run`.
"""

from areval.utils.serialization import reconstruct_test_result, reconstruct_run
from areval.test_case import (
    TestCase,
    AgentOutput,
    TestResult,
    EvaluationRun,
    TestStatus,
)


class TestReconstructTestResult:
    """Deserialisation back to TestResult."""

    def test_roundtrip_passed(self):
        tc = TestCase(id="t1", name="case1", input="hello", expected_output="world")
        ao = AgentOutput(output="hello world", latency_ms=50.0, cost_usd=0.01)
        original = TestResult(
            test_case=tc,
            agent_output=ao,
            status=TestStatus.PASSED,
            scores={"exact_match": 1.0},
            overall_score=1.0,
            threshold=0.7,
            judge_reasoning="perfect",
        )
        d = original.to_dict()
        restored = reconstruct_test_result(d)
        assert restored.test_case.id == "t1"
        assert restored.test_case.name == "case1"
        assert restored.agent_output.output == "hello world"
        assert restored.agent_output.latency_ms == 50.0
        assert restored.agent_output.cost_usd == 0.01
        assert restored.status == TestStatus.PASSED
        assert restored.overall_score == 1.0
        assert restored.passed is True
        assert restored.judge_reasoning == "perfect"

    def test_roundtrip_failed(self):
        tc = TestCase(id="t2", name="case2", input="q", expected_output="a")
        ao = AgentOutput(output="wrong")
        original = TestResult(
            test_case=tc,
            agent_output=ao,
            status=TestStatus.FAILED,
            scores={"exact_match": 0.0},
            overall_score=0.0,
            threshold=0.7,
        )
        d = original.to_dict()
        restored = reconstruct_test_result(d)
        assert restored.status == TestStatus.FAILED
        assert restored.overall_score == 0.0
        assert restored.passed is False

    def test_roundtrip_error(self):
        tc = TestCase(id="t3", name="err", input="bad")
        ao = AgentOutput(output="")
        original = TestResult(
            test_case=tc,
            agent_output=ao,
            status=TestStatus.ERROR,
            scores={},
            overall_score=0.0,
            error_message="something crashed",
        )
        d = original.to_dict()
        restored = reconstruct_test_result(d)
        assert restored.status == TestStatus.ERROR
        assert restored.error_message == "something crashed"

    def test_regression_fields(self):
        tc = TestCase(id="t4", name="reg", input="x", expected_output="y")
        ao = AgentOutput(output="z")
        original = TestResult(
            test_case=tc,
            agent_output=ao,
            overall_score=0.6,
            threshold=0.7,
            baseline_score=0.9,
            regression_delta=-0.3,
            is_regression=True,
        )
        d = original.to_dict()
        restored = reconstruct_test_result(d)
        assert restored.is_regression is True
        assert restored.regression_delta == -0.3
        assert restored.baseline_score == 0.9

    def test_missing_optional_fields(self):
        """Fields not present in the dict should get sensible defaults."""
        minimal = {
            "test_case": {
                "id": "m1",
                "name": "minimal",
                "input": "hi",
            },
            "agent_output": {"output": "ok"},
        }
        restored = reconstruct_test_result(minimal)
        assert restored.status == TestStatus.PASSED
        assert restored.overall_score == 0.0
        assert restored.threshold == 0.7
        assert restored.execution_time_ms == 0.0
        assert restored.is_regression is False


class TestReconstructRun:
    """Deserialisation back to EvaluationRun."""

    def test_roundtrip_empty(self):
        run = EvaluationRun(name="empty-run", description="nothing")
        d = run.to_dict()
        restored = reconstruct_run(d)
        assert restored.name == "empty-run"
        assert restored.description == "nothing"
        assert restored.total_cases == 0

    def test_roundtrip_with_results(self):
        tc = TestCase(id="c1", name="c1", input="q", expected_output="a")
        ao = AgentOutput(output="a", latency_ms=10.0)
        tr = TestResult(
            test_case=tc,
            agent_output=ao,
            status=TestStatus.PASSED,
            scores={"exact": 1.0},
            overall_score=1.0,
            threshold=0.7,
        )
        run = EvaluationRun(name="one-case", description="single")
        run.test_results = [tr]
        run._compute_aggregates()
        d = run.to_dict()
        restored = reconstruct_run(d)
        assert restored.name == "one-case"
        assert restored.total_cases == 1
        assert restored.passed_cases == 1
        assert restored.failed_cases == 0

    def test_with_skipped_and_timed_out(self):
        tc_pass = TestCase(id="p1", name="pass", input="a", expected_output="a")
        ao_pass = AgentOutput(output="a")
        tr_pass = TestResult(
            test_case=tc_pass, agent_output=ao_pass,
            status=TestStatus.PASSED, overall_score=1.0, threshold=0.7,
        )

        tc_skip = TestCase(id="s1", name="skip", input="b")
        ao_skip = AgentOutput(output="")
        tr_skip = TestResult(
            test_case=tc_skip, agent_output=ao_skip,
            status=TestStatus.SKIPPED, overall_score=0.0,
        )

        tc_to = TestCase(id="to1", name="timeout", input="c")
        ao_to = AgentOutput(output="")
        tr_to = TestResult(
            test_case=tc_to, agent_output=ao_to,
            status=TestStatus.TIMEOUT, overall_score=0.0,
        )

        run = EvaluationRun(name="mixed")
        run.test_results = [tr_pass, tr_skip, tr_to]
        run._compute_aggregates()
        d = run.to_dict()
        restored = reconstruct_run(d)
        assert restored.total_cases == 3
        assert restored.passed_cases == 1
        assert restored.skipped_cases == 1
        assert restored.timed_out_cases == 1
        assert restored.failed_cases == 0
