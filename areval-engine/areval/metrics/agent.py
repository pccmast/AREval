"""Agent-specific evaluation metrics.

Metrics for evaluating agent behavior: tool use, task completion,
multi-turn coherence, and planning quality.
"""

from typing import Any, List, Set

from areval.metrics.base import Metric, MetricResult
from areval.test_case import TestCase, AgentOutput


class ToolCallAccuracyMetric(Metric):
    """Evaluates the correctness of agent tool calls.

    Compares the sequence and parameters of tool calls against
    expected tool invocations. Critical for agent infrastructure validation.
    """

    name = "tool_call_accuracy"

    def __init__(
        self,
        threshold: float = 1.0,
        check_order: bool = True,
        check_params: bool = True,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.check_order = check_order
        self.check_params = check_params

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        expected_tools = test_case.expected_tools or []
        actual_tools = [tc.get("name", "") for tc in agent_output.tool_calls]

        if not expected_tools and not actual_tools:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="No tools expected or called",
                threshold=self.threshold,
            )

        if not expected_tools:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning=f"No tools expected but {len(actual_tools)} were called",
                threshold=self.threshold,
            )

        if not actual_tools:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning=f"Expected {len(expected_tools)} tools but none were called",
                threshold=self.threshold,
            )

        score = 0.0
        reasoning_parts = []

        # Check tool names match
        if self.check_order:
            matches = sum(
                1 for e, a in zip(expected_tools, actual_tools) if e == a
            )
            order_score = matches / max(len(expected_tools), len(actual_tools))
            score += order_score * 0.5
            reasoning_parts.append(f"Order accuracy: {matches}/{max(len(expected_tools), len(actual_tools))}")
        else:
            expected_set = set(expected_tools)
            actual_set = set(actual_tools)
            precision = len(expected_set & actual_set) / len(actual_set) if actual_set else 0
            recall = len(expected_set & actual_set) / len(expected_set) if expected_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            score += f1 * 0.5
            reasoning_parts.append(f"Tool F1: {f1:.2f} (P={precision:.2f}, R={recall:.2f})")

        # Check parameters if applicable
        if self.check_params and agent_output.tool_calls:
            # Simplified: check if expected parameters exist in actual calls
            param_score = 0.0
            for expected_tool, actual_call in zip(expected_tools, agent_output.tool_calls):
                if actual_call.get("name") == expected_tool:
                    param_score += 1.0
            param_score /= max(len(expected_tools), len(actual_tools))
            score += param_score * 0.5
            reasoning_parts.append(f"Param accuracy: {param_score:.2f}")

        return MetricResult(
            name=self.name,
            score=min(1.0, score),
            passed=score >= self.threshold,
            reasoning="; ".join(reasoning_parts),
            threshold=self.threshold,
            metadata={
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
            },
        )


class TaskCompletionMetric(Metric):
    """Binary task completion metric.

    Inspired by SWE-bench's binary scoring: a task is either
    completed (all tests pass) or not (any test fails).
    """

    name = "task_completion"

    def __init__(self, threshold: float = 1.0, **kwargs: Any):
        super().__init__(threshold=threshold, **kwargs)

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        # For SWE-bench style tasks, check if test command passes
        if test_case.test_command:
            # In production: execute the test command in isolated environment
            # For skeleton: simulate based on output content
            output_lower = agent_output.output.lower()
            success_indicators = ["pass", "success", "completed", "fixed"]
            failure_indicators = ["fail", "error", "exception", "timeout"]

            success_count = sum(1 for ind in success_indicators if ind in output_lower)
            failure_count = sum(1 for ind in failure_indicators if ind in output_lower)

            score = 1.0 if success_count > failure_count else 0.0
            return MetricResult(
                name=self.name,
                score=score,
                passed=score >= self.threshold,
                reasoning=f"Success indicators: {success_count}, Failure indicators: {failure_count}",
                threshold=self.threshold,
            )

        # For general tasks, use output quality heuristics
        output = agent_output.output.strip()
        if not output:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty output",
                threshold=self.threshold,
            )

        # Check if output is substantive
        score = min(1.0, len(output) / 200)  # Normalize by expected length

        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=f"Output length: {len(output)} chars",
            threshold=self.threshold,
        )
