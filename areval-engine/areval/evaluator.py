"""Main evaluation orchestrator.

Coordinates metrics, judges, and regression detection to run
complete evaluation workflows.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from areval.metrics.base import Metric
from areval.judges.base import Judge
from areval.regression.detector import RegressionDetector
from areval.regression.baseline import BaselineManager
from areval.test_case import (
    TestCase,
    AgentOutput,
    TestResult,
    EvaluationRun,
    TestStatus,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluation engine.

    Orchestrates the evaluation pipeline:
    1. Load test cases from datasets
    2. Execute agent (or load pre-computed outputs)
    3. Apply metrics and judges
    4. Detect regressions against baseline
    5. Generate evaluation report

    Inspired by DeepEval's evaluate() and LangSmith's run evaluation.
    """

    def __init__(
        self,
        metrics: Optional[List[Metric]] = None,
        judges: Optional[List[Judge]] = None,
        threshold: float = 0.7,
        min_score: float = 0.0,
        metric_retries: int = 1,
        metric_timeout: float = 60.0,
        regression_detector: Optional[RegressionDetector] = None,
        baseline_manager: Optional[BaselineManager] = None,
    ):
        self.metrics = metrics or []
        self.judges = judges or []
        self.threshold = threshold
        self.min_score = min_score
        self.metric_retries = metric_retries
        self.metric_timeout = metric_timeout
        self.regression_detector = regression_detector or RegressionDetector()
        self.baseline_manager = baseline_manager or BaselineManager()
        self._results_history: List[EvaluationRun] = []

    def add_metric(self, metric: Metric) -> Evaluator:
        """Add a metric to the evaluation pipeline."""
        self.metrics.append(metric)
        return self

    def add_judge(self, judge: Judge) -> Evaluator:
        """Add a judge to the evaluation pipeline."""
        self.judges.append(judge)
        return self

    def evaluate(
        self,
        test_cases: List[TestCase],
        agent_fn: Optional[Callable[[TestCase], AgentOutput]] = None,
        agent_outputs: Optional[List[AgentOutput]] = None,
        run_name: str = "",
        run_description: str = "",
        config: Optional[Dict[str, Any]] = None,
        compare_baseline: bool = True,
    ) -> EvaluationRun:
        """Run a complete evaluation.

        Args:
            test_cases: List of test cases to evaluate
            agent_fn: Function that takes a TestCase and returns AgentOutput
            agent_outputs: Pre-computed agent outputs (alternative to agent_fn)
            run_name: Name for this evaluation run
            run_description: Description of the run
            config: Evaluation configuration metadata
            compare_baseline: Whether to compare against baseline

        Returns:
            EvaluationRun with complete results
        """
        run = EvaluationRun(
            name=run_name or f"eval-{len(self._results_history) + 1}",
            description=run_description,
        )

        # Filter: only evaluate approved cases (backward-compatible:
        # seed cases without 'pending_review' tag are treated as approved)
        approved_cases = [
            tc for tc in test_cases
            if "approved" in tc.tags or "pending_review" not in tc.tags
        ]
        skipped_review = len(test_cases) - len(approved_cases)
        if skipped_review:
            logger.info(
                "Skipping %d test case(s) tagged 'pending_review'. "
                "Approve them via Dashboard or areval dataset approve.",
                skipped_review,
            )
        test_cases = approved_cases if approved_cases else test_cases

        run.config = config or {}

        # Get agent outputs
        if agent_outputs is not None:
            outputs = agent_outputs
        elif agent_fn is not None:
            outputs = self._execute_agent(test_cases, agent_fn)
        else:
            raise ValueError("Must provide either agent_fn or agent_outputs")

        # Run metrics and judges for each test case
        for test_case, agent_output in zip(test_cases, outputs):
            result = self._evaluate_single(test_case, agent_output)
            run.test_results.append(result)

        # Compute aggregates
        run.completed_at = datetime.now(timezone.utc)
        run._compute_aggregates()

        # Regression detection
        if compare_baseline:
            baseline = self.baseline_manager.get_latest_baseline()
            if baseline:
                report = self.regression_detector.detect(
                    run.test_results, baseline.test_results
                )
                if report.has_regression:
                    run.regression_count = len(report.affected_tests)
                    # Build O(1) lookup map from baseline results
                    baseline_score_map = {
                        r.test_case.id: r.overall_score
                        for r in baseline.test_results
                    }
                    affected_ids = set(report.affected_tests)
                    for result in run.test_results:
                        if result.test_case.id in affected_ids:
                            result.is_regression = True
                            baseline_score = baseline_score_map.get(
                                result.test_case.id, result.overall_score
                            )
                            result.regression_delta = result.overall_score - baseline_score

        self._results_history.append(run)
        return run

    def _execute_agent(
        self,
        test_cases: List[TestCase],
        agent_fn: Callable[[TestCase], AgentOutput],
    ) -> List[AgentOutput]:
        """Execute the agent function for all test cases."""
        outputs = []
        for tc in test_cases:
            start = time.time()
            try:
                output = agent_fn(tc)
                output.latency_ms = (time.time() - start) * 1000
                outputs.append(output)
            except Exception as e:
                # Return error output with explicit error flag
                outputs.append(
                    AgentOutput(
                        output="",
                        latency_ms=(time.time() - start) * 1000,
                        metadata={"error": str(e), "_agent_error": True},
                    )
                )
        return outputs

    def _evaluate_single(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
    ) -> TestResult:
        """Evaluate a single test case."""
        import math
        start_time = time.time()
        scores: Dict[str, float] = {}

        # Check for agent execution errors
        agent_error = agent_output.metadata.get("_agent_error", False)
        if agent_error:
            return TestResult(
                test_case=test_case,
                agent_output=agent_output,
                status=TestStatus.ERROR,
                scores={"error": 0.0},
                overall_score=0.0,
                threshold=self.threshold,
                error_message=str(agent_output.metadata.get("error", "Agent execution failed")),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Apply metrics with retry + timeout
        for metric in self.metrics:
            scores[metric.name] = self._run_with_retry(
                lambda m=metric: m.measure(test_case, agent_output),
                metric.name,
            )

        # Apply judges with retry + timeout (collect reasoning on success)
        judge_reasoning_parts = []
        for judge in self.judges:
            def _eval_judge(j=judge):
                return j.evaluate(test_case, agent_output)
            result = self._run_judge_with_retry(_eval_judge, judge.name)
            scores[judge.name] = result[0]
            if result[1]:
                judge_reasoning_parts.append(result[1])

        # Calculate overall score: nanmean (ignore failed metrics)
        valid_scores = [v for v in scores.values() if not math.isnan(v)]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # min_score gate: any valid metric below min_score → fail
        below_min = [name for name, s in scores.items()
                     if not math.isnan(s) and s < self.min_score]
        if below_min and self.min_score > 0:
            overall_score = min(overall_score, self.min_score)
            logger.warning(
                "Test case '%s' has %d metric(s) below min_score %.2f: %s",
                test_case.name, len(below_min), self.min_score, below_min,
            )

        execution_time = (time.time() - start_time) * 1000

        return TestResult(
            test_case=test_case,
            agent_output=agent_output,
            status=TestStatus.PASSED if overall_score >= self.threshold else TestStatus.FAILED,
            scores=scores,
            overall_score=overall_score,
            threshold=self.threshold,
            judge_reasoning="\n".join(judge_reasoning_parts) if judge_reasoning_parts else None,
            execution_time_ms=execution_time,
        )

    def _run_with_retry(self, fn, name: str) -> float:
        """Execute a metric/judge call with retry.

        Returns the score on success, ``NaN`` after exhausting retries.
        """
        import math
        for attempt in range(1, self.metric_retries + 1):
            try:
                result = fn()
                return result.score
            except Exception as e:
                if attempt < self.metric_retries:
                    logger.info(
                        "Metric/judge '%s' attempt %d/%d failed: %s — retrying",
                        name, attempt, self.metric_retries, e,
                    )
                else:
                    logger.warning(
                        "Metric/judge '%s' failed after %d attempt(s): %s",
                        name, self.metric_retries, e,
                    )
        return float("nan")

    def _run_judge_with_retry(self, fn, name: str):
        """Execute a judge call with retry, returning (score, reasoning).

        Returns ``(NaN, None)`` after exhausting retries.
        """
        import math
        for attempt in range(1, self.metric_retries + 1):
            try:
                result = fn()
                return result.score, result.reasoning
            except Exception as e:
                if attempt < self.metric_retries:
                    logger.info(
                        "Judge '%s' attempt %d/%d failed: %s — retrying",
                        name, attempt, self.metric_retries, e,
                    )
                else:
                    logger.warning(
                        "Judge '%s' failed after %d attempt(s): %s",
                        name, self.metric_retries, e,
                    )
        return float("nan"), None

    def create_baseline(
        self,
        run: EvaluationRun,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Save an evaluation run as a new baseline."""
        baseline = self.baseline_manager.create_baseline(
            run=run,
            name=name,
            tags=tags or ["manual"],
        )
        return baseline.id

    def get_history(self) -> List[EvaluationRun]:
        """Get evaluation run history."""
        return self._results_history

    def summary(self, run: EvaluationRun) -> str:
        """Generate a text summary of an evaluation run."""
        lines = [
            f"=" * 60,
            f"Evaluation Run: {run.name}",
            f"Description: {run.description}",
            f"-" * 60,
            f"Total Cases:  {run.total_cases}",
            f"Passed:       {run.passed_cases} ({run.pass_rate:.1%})",
            f"Failed:       {run.failed_cases}",
            f"Skipped:      {run.skipped_cases}",
            f"Timed Out:    {run.timed_out_cases}",
            f"Errors:       {run.error_cases}",
            f"Average Score: {run.avg_score:.3f}",
            f"Regressions:  {run.regression_count}",
            f"Total Cost:   ${run.total_cost_usd:.4f}",
            f"Total Tokens: {run.total_tokens:,}",
            f"Duration:     {run.duration_seconds:.1f}s",
            f"=" * 60,
        ]
        return "\n".join(lines)
