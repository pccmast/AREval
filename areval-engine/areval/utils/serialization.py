"""Serialization utilities for reconstructing evaluation objects from dicts.

Centralises the TestResult and EvaluationRun reconstruction logic that was
previously duplicated across areval-cli, BaselineManager, and JSONReporter.
"""

from __future__ import annotations

from typing import Any, Dict, List

from areval.test_case import (
    TestCase,
    AgentOutput,
    TestResult,
    EvaluationRun,
    TestStatus,
)


def reconstruct_test_result(data: Dict[str, Any]) -> TestResult:
    """Reconstruct a TestResult from its serialized dict.

    This is the single source of truth for deserialising individual test
    results.  Previously duplicated in:
      * areval_cli.main._reconstruct_run (inline)
      * areval.regression.baseline._reconstruct_test_result
    """
    tc_data = data["test_case"]
    ao_data = data["agent_output"]

    tc = TestCase.from_dict(tc_data)
    ao = AgentOutput(
        output=ao_data.get("output", ""),
        tool_calls=ao_data.get("tool_calls", []),
        latency_ms=ao_data.get("latency_ms", 0.0),
        token_usage=ao_data.get("token_usage", {}),
        cost_usd=ao_data.get("cost_usd", 0.0),
        trace_id=ao_data.get("trace_id"),
    )

    return TestResult(
        test_case=tc,
        agent_output=ao,
        status=TestStatus(data.get("status", "passed")),
        scores=data.get("scores", {}),
        overall_score=data.get("overall_score", 0.0),
        threshold=data.get("threshold", 0.7),
        error_message=data.get("error_message"),
        judge_reasoning=data.get("judge_reasoning"),
        execution_time_ms=data.get("execution_time_ms", 0.0),
        baseline_score=data.get("baseline_score"),
        regression_delta=data.get("regression_delta"),
        is_regression=data.get("is_regression", False),
    )


def reconstruct_run(data: Dict[str, Any]) -> EvaluationRun:
    """Reconstruct an EvaluationRun from its serialized dict.

    This is the single source of truth for deserialising complete evaluation
    runs.  Previously duplicated in:
      * areval_cli.main._reconstruct_run
    """
    results: List[TestResult] = [
        reconstruct_test_result(r_data) for r_data in data.get("test_results", [])
    ]

    run = EvaluationRun(
        name=data.get("name", "loaded"),
        description=data.get("description", ""),
        config=data.get("config", {}),
    )
    run.test_results = results
    run.total_cases = data.get("total_cases", len(results))
    run._compute_aggregates()
    return run
