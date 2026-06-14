"""Tests for the Evaluator orchestrator (Sprint 1.5)."""

import pytest

from areval.evaluator import Evaluator
from areval.test_case import TestCase, AgentOutput, EvaluationRun, TestStatus
from areval.metrics import ExactMatchMetric, ContainsMetric, SemanticSimilarityMetric
from areval.judges import LLMJudge


class TestEvaluator:
    """Integration-style tests for the evaluation engine."""

    def _make_case(self, name: str, expected: str = "correct") -> TestCase:
        return TestCase(name=name, input=name, expected_output=expected)

    # ------------------------------------------------------------------
    # Basic agent execution
    # ------------------------------------------------------------------

    def test_evaluate_with_agent_fn(self) -> None:
        evaluator = Evaluator()
        evaluator.add_metric(ExactMatchMetric())

        cases = [self._make_case("a", "hello"), self._make_case("b", "world")]

        def agent(tc: TestCase) -> AgentOutput:
            return AgentOutput(output=tc.expected_output or "")

        run = evaluator.evaluate(cases, agent_fn=agent)

        assert isinstance(run, EvaluationRun)
        assert run.total_cases == 2
        assert run.pass_rate == 1.0

    def test_evaluate_with_outputs(self) -> None:
        evaluator = Evaluator()
        evaluator.add_metric(ExactMatchMetric())

        cases = [self._make_case("a", "hello")]
        outputs = [AgentOutput(output="hello")]

        run = evaluator.evaluate(cases, agent_outputs=outputs)

        assert run.pass_rate == 1.0

    # ------------------------------------------------------------------
    # Metrics and scoring
    # ------------------------------------------------------------------

    def test_evaluate_metrics_applied(self) -> None:
        evaluator = Evaluator()
        evaluator.add_metric(ExactMatchMetric())
        evaluator.add_metric(ContainsMetric())

        cases = [self._make_case("test", "foo bar baz")]
        outputs = [AgentOutput(output="foo bar baz")]

        run = evaluator.evaluate(cases, agent_outputs=outputs)
        result = run.test_results[0]

        assert "exact_match" in result.scores
        assert "contains" in result.scores
        assert result.scores["exact_match"] == 1.0

    def test_evaluate_judges_applied(self) -> None:
        evaluator = Evaluator()
        evaluator.add_metric(ExactMatchMetric())
        evaluator.add_judge(LLMJudge(provider="mock", threshold=0.5))

        cases = [self._make_case("test", "the answer")]
        outputs = [AgentOutput(output="the answer")]

        run = evaluator.evaluate(cases, agent_outputs=outputs)
        result = run.test_results[0]

        assert "llm_judge" in result.scores
        assert 0.0 <= result.scores["llm_judge"] <= 1.0

    def test_overall_score_average(self) -> None:
        evaluator = Evaluator(threshold=0.5)
        evaluator.add_metric(ExactMatchMetric())  # score = 1.0 (match)
        # Semantic uses offline → score is computed deterministically

        cases = [self._make_case("test", "hello world")]
        outputs = [AgentOutput(output="hello world")]

        run = evaluator.evaluate(cases, agent_outputs=outputs)
        result = run.test_results[0]

        # exact_match should be 1.0, so overall >= 0.5
        assert result.overall_score >= 0.5
        assert result.passed

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_agent_fn_exception(self) -> None:
        evaluator = Evaluator()
        evaluator.add_metric(ExactMatchMetric())

        def failing_agent(tc: TestCase) -> AgentOutput:
            raise RuntimeError("agent crashed")

        cases = [self._make_case("a")]

        # Should not throw — evaluator catches agent errors
        run = evaluator.evaluate(cases, agent_fn=failing_agent)

        assert run.test_results[0].status == TestStatus.ERROR

    def test_empty_test_cases(self) -> None:
        evaluator = Evaluator()
        evaluator.add_metric(ExactMatchMetric())

        run = evaluator.evaluate([], agent_outputs=[])

        assert run.total_cases == 0
        assert run.pass_rate == 0.0

    # ------------------------------------------------------------------
    # Missing inputs
    # ------------------------------------------------------------------

    def test_missing_agent_fn_and_outputs(self) -> None:
        evaluator = Evaluator()
        with pytest.raises(ValueError, match="agent_fn or agent_outputs"):
            evaluator.evaluate([], run_name="bad")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def test_summary_format(self) -> None:
        evaluator = Evaluator()
        evaluator.add_metric(ExactMatchMetric())

        cases = [self._make_case("a", "yes")]
        outputs = [AgentOutput(output="yes")]

        run = evaluator.evaluate(cases, agent_outputs=outputs)

        text = evaluator.summary(run)
        assert "Total Cases:" in text
        assert "Passed:" in text
        assert "Average Score:" in text
        assert run.name in text

    # ------------------------------------------------------------------
    # Chain API
    # ------------------------------------------------------------------

    def test_chain_api(self) -> None:
        evaluator = (
            Evaluator()
            .add_metric(ExactMatchMetric())
            .add_metric(ContainsMetric())
            .add_judge(LLMJudge(provider="mock", threshold=0.5))
        )
        assert len(evaluator.metrics) == 2
        assert len(evaluator.judges) == 1

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    def test_baseline_comparison(self) -> None:
        evaluator = Evaluator()
        evaluator.add_metric(ExactMatchMetric())

        cases = [self._make_case("a", "correct")]
        outputs = [AgentOutput(output="correct")]

        # First run — create baseline
        run1 = evaluator.evaluate(
            cases, agent_outputs=outputs, run_name="baseline-run"
        )
        evaluator.create_baseline(run1, name="v1")

        # Second run — should compare against baseline
        run2 = evaluator.evaluate(
            cases, agent_outputs=outputs, run_name="current-run",
        )

        assert isinstance(run2, EvaluationRun)
