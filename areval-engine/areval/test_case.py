"""Test case and result data models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TestStatus(str, Enum):
    """Status of a test case execution."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TestCase:
    """A single test case for agent evaluation.

    Represents an input scenario with expected behavior criteria.
    Compatible with SWE-bench-style evaluation patterns.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    input: str = ""
    expected_output: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    timeout_seconds: float = 120.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    # SWE-bench style fields
    task_id: Optional[str] = None
    repository: Optional[str] = None
    base_commit: Optional[str] = None
    test_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "input": self.input,
            "expected_output": self.expected_output,
            "expected_tools": self.expected_tools,
            "context": self.context,
            "metadata": self.metadata,
            "tags": self.tags,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "task_id": self.task_id,
            "repository": self.repository,
            "base_commit": self.base_commit,
            "test_command": self.test_command,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestCase:
        data = data.copy()
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentOutput:
    """The output produced by an agent for a given test case."""

    output: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    thinking: Optional[str] = None
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Complete result of evaluating a test case."""

    test_case: TestCase
    agent_output: AgentOutput
    status: TestStatus = TestStatus.PASSED
    scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    threshold: float = 0.7
    passed: bool = False
    error_message: Optional[str] = None
    judge_reasoning: Optional[str] = None
    execution_time_ms: float = 0.0
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    # Regression tracking
    baseline_score: Optional[float] = None
    regression_delta: Optional[float] = None
    is_regression: bool = False

    def __post_init__(self):
        self.passed = self.overall_score >= self.threshold and self.status == TestStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case": self.test_case.to_dict(),
            "agent_output": {
                "output": self.agent_output.output,
                "tool_calls": self.agent_output.tool_calls,
                "latency_ms": self.agent_output.latency_ms,
                "token_usage": self.agent_output.token_usage,
                "cost_usd": self.agent_output.cost_usd,
                "trace_id": self.agent_output.trace_id,
            },
            "status": self.status.value,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "threshold": self.threshold,
            "passed": self.passed,
            "error_message": self.error_message,
            "judge_reasoning": self.judge_reasoning,
            "execution_time_ms": self.execution_time_ms,
            "evaluated_at": self.evaluated_at.isoformat(),
            "baseline_score": self.baseline_score,
            "regression_delta": self.regression_delta,
            "is_regression": self.is_regression,
        }


@dataclass
class EvaluationRun:
    """A complete evaluation run across multiple test cases."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    test_results: List[TestResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    # Aggregate metrics
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    error_cases: int = 0
    avg_score: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    regression_count: int = 0

    def __post_init__(self):
        self._compute_aggregates()

    def _compute_aggregates(self):
        if not self.test_results:
            return
        self.total_cases = len(self.test_results)
        self.passed_cases = sum(1 for r in self.test_results if r.passed)
        self.failed_cases = sum(1 for r in self.test_results if not r.passed and r.status != TestStatus.ERROR)
        self.error_cases = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        self.avg_score = sum(r.overall_score for r in self.test_results) / self.total_cases
        self.total_cost_usd = sum(r.agent_output.cost_usd for r in self.test_results)
        self.total_tokens = sum(
            sum(r.agent_output.token_usage.values()) for r in self.test_results
        )
        self.regression_count = sum(1 for r in self.test_results if r.is_regression)

    @property
    def pass_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "error_cases": self.error_cases,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "regression_count": self.regression_count,
            "duration_seconds": self.duration_seconds,
            "test_results": [r.to_dict() for r in self.test_results],
        }
