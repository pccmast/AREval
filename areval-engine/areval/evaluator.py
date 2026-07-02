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
        regression_detector: Optional[RegressionDetector] = None,
        baseline_manager: Optional[BaselineManager] = None,
    ):
        self.metrics = metrics or []
        self.judges = judges or []
        self.threshold = threshold
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

        # Apply metrics
        for metric in self.metrics:
            try:
                result = metric.measure(test_case, agent_output)
                scores[result.name] = result.score
            except Exception as e:
                logger.warning(
                    "Metric '%s' failed for test case '%s': %s",
                    metric.name, test_case.name, e,
                )
                scores[metric.name] = 0.0

        # Apply judges
        judge_reasoning_parts = []
        for judge in self.judges:
            try:
                result = judge.evaluate(test_case, agent_output)
                scores[judge.name] = result.score
                if result.reasoning:
                    judge_reasoning_parts.append(result.reasoning)
            except Exception as e:
                logger.warning(
                    "Judge '%s' failed for test case '%s': %s",
                    judge.name, test_case.name, e,
                )
                scores[judge.name] = 0.0

        # Calculate overall score (average of all scores)
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0

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
