"""Deterministic accuracy metrics."""

import re
from typing import Any, Optional

from areval.metrics.base import Metric, MetricResult
from areval.test_case import TestCase, AgentOutput


class ExactMatchMetric(Metric):
    """Exact string match metric.

    Simple but unforgiving — useful for structured output validation.
    Inspired by SWE-bench binary scoring pattern.
    """

    name = "exact_match"

    def __init__(
        self,
        threshold: float = 1.0,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        if test_case.expected_output is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No expected_output set in test case",
                threshold=self.threshold,
            )

        expected = test_case.expected_output
        actual = agent_output.output

        if self.strip_whitespace:
            expected = expected.strip()
            actual = actual.strip()

        if not self.case_sensitive:
            expected = expected.lower()
            actual = actual.lower()

        score = 1.0 if expected == actual else 0.0

        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=f"Expected: '{test_case.expected_output[:100]}...', Got: '{agent_output.output[:100]}...'",
            threshold=self.threshold,
        )


class ContainsMetric(Metric):
    """Check if output contains expected substring(s).

    Useful for verifying specific keywords, facts, or structured elements
    are present in the agent response.
    """

    name = "contains"

    def __init__(
        self,
        threshold: float = 1.0,
        all_required: bool = True,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.all_required = all_required

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        expected = test_case.expected_output
        if expected is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No expected_output set",
                threshold=self.threshold,
            )

        actual = agent_output.output.lower()
        expected_parts = [p.strip().lower() for p in expected.split("|")]

        matches = sum(1 for part in expected_parts if part in actual)

        if self.all_required:
            score = 1.0 if matches == len(expected_parts) else matches / len(expected_parts)
        else:
            score = matches / len(expected_parts) if expected_parts else 0.0

        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=f"Matched {matches}/{len(expected_parts)} required parts",
            threshold=self.threshold,
        )


class RegexMatchMetric(Metric):
    """Regex pattern matching metric.

    Flexible validation for structured outputs like JSON, code blocks,
    or any pattern-validated content.
    """

    name = "regex_match"

    def __init__(
        self,
        threshold: float = 1.0,
        pattern: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.pattern = pattern
        self._compiled = re.compile(pattern) if pattern else None

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        pattern = self.pattern or test_case.expected_output
        if not pattern:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No regex pattern provided",
                threshold=self.threshold,
            )

        try:
            compiled = re.compile(pattern, re.DOTALL)
            match = compiled.search(agent_output.output)
            score = 1.0 if match else 0.0

            return MetricResult(
                name=self.name,
                score=score,
                passed=score >= self.threshold,
                reasoning=f"Pattern {'matched' if match else 'did not match'}",
                threshold=self.threshold,
                metadata={"groups": match.groupdict() if match else None},
            )
        except re.error as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning=f"Invalid regex pattern: {e}",
                threshold=self.threshold,
            )
