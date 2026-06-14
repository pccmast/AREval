"""Regression detection algorithms.

Statistical methods for detecting significant performance degradation
between evaluation runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from areval.test_case import TestResult, EvaluationRun


@dataclass
class RegressionReport:
    """Detailed regression analysis report."""

    has_regression: bool = False
    confidence: float = 0.0
    affected_tests: List[str] = field(default_factory=list)
    score_delta: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_regression": self.has_regression,
            "confidence": self.confidence,
            "affected_tests": self.affected_tests,
            "score_delta": self.score_delta,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "details": self.details,
        }


class RegressionDetector:
    """Detects performance regressions between evaluation runs.

    Uses statistical methods:
    - Paired t-test for score differences
    - Effect size (Cohen's d) for practical significance
    - Threshold-based alerts for critical metrics

    Inspired by SWE-bench and CI/CD regression testing patterns.
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        min_effect_size: float = 0.2,
        absolute_threshold: float = 0.05,
    ):
        self.significance_threshold = significance_threshold
        self.min_effect_size = min_effect_size
        self.absolute_threshold = absolute_threshold

    def detect(
        self,
        current_results: List[TestResult],
        baseline_results: List[TestResult],
    ) -> RegressionReport:
        """Compare current results against baseline and detect regressions.

        Args:
            current_results: Latest evaluation results
            baseline_results: Previous baseline results to compare against

        Returns:
            RegressionReport with detailed analysis
        """
        report = RegressionReport()

        if not baseline_results or not current_results:
            return report

        # Build lookup by test case ID
        baseline_map = {r.test_case.id: r for r in baseline_results}
        current_map = {r.test_case.id: r for r in current_results}

        # Find common test cases
        common_ids = set(baseline_map.keys()) & set(current_map.keys())

        if len(common_ids) < 2:
            report.details.append({
                "warning": "Insufficient common test cases for statistical comparison",
                "common_count": len(common_ids),
            })
            return report

        # Collect paired scores
        baseline_scores = []
        current_scores = []
        deltas = []
        affected = []

        for tc_id in common_ids:
            b = baseline_map[tc_id]
            c = current_map[tc_id]
            baseline_scores.append(b.overall_score)
            current_scores.append(c.overall_score)
            delta = c.overall_score - b.overall_score
            deltas.append(delta)

            if delta < -self.absolute_threshold:
                affected.append({
                    "test_id": tc_id,
                    "test_name": c.test_case.name,
                    "baseline": b.overall_score,
                    "current": c.overall_score,
                    "delta": delta,
                })

        # Statistical tests
        if len(deltas) >= 2:
            report.score_delta = float(np.mean(deltas))
            report.p_value = self._paired_ttest(baseline_scores, current_scores)
            report.effect_size = self._cohens_d(baseline_scores, current_scores)

            # Determine regression
            report.has_regression = (
                report.p_value < self.significance_threshold
                and report.effect_size > self.min_effect_size
                and report.score_delta < -self.absolute_threshold
            )

            report.confidence = 1.0 - report.p_value
            report.affected_tests = [a["test_id"] for a in affected]
            report.details = affected

        return report

    def detect_run_regression(
        self,
        current_run: EvaluationRun,
        baseline_run: EvaluationRun,
    ) -> RegressionReport:
        """Compare two complete evaluation runs."""
        return self.detect(current_run.test_results, baseline_run.test_results)

    def _paired_ttest(self, a: List[float], b: List[float]) -> float:
        """Calculate paired t-test p-value using scipy's t-distribution CDF.

        Uses scipy.stats.t.cdf for accurate p-value computation, which is
        essential for the small sample sizes (n >= 2) common in agent
        evaluation regression testing.
        """
        if len(a) != len(b) or len(a) < 2:
            return 1.0

        from scipy import stats

        diffs = np.array(b) - np.array(a)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)

        if std_diff == 0:
            return 0.0 if mean_diff != 0 else 1.0

        n = len(diffs)
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        df = n - 1

        # Two-tailed p-value from Student's t-distribution
        p_value = 2.0 * stats.t.cdf(-abs(t_stat), df)
        return float(p_value)

    def _cohens_d(self, a: List[float], b: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(a) != len(b) or len(a) < 2:
            return 0.0

        diffs = np.array(b) - np.array(a)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)

        if std_diff == 0:
            return 0.0

        return float(abs(mean_diff / std_diff))

    def classify_severity(self, report: RegressionReport) -> str:
        """Classify regression severity."""
        if not report.has_regression:
            return "none"

        if report.effect_size > 0.8 or report.score_delta < -0.2:
            return "critical"
        elif report.effect_size > 0.5 or report.score_delta < -0.1:
            return "major"
        else:
            return "minor"
