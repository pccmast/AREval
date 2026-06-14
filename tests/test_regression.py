"""Tests for regression detection."""

import pytest

from areval.test_case import TestCase, AgentOutput, TestResult, TestStatus
from areval.regression.detector import RegressionDetector, RegressionReport


class TestRegressionDetector:
    def test_no_regression(self):
        detector = RegressionDetector()

        # Baseline and current are similar
        baseline = self._create_results([0.8, 0.85, 0.9])
        current = self._create_results([0.82, 0.84, 0.88])

        report = detector.detect(current, baseline)
        assert not report.has_regression

    def test_detects_regression(self):
        detector = RegressionDetector(
            significance_threshold=0.1,
            min_effect_size=0.1,
            absolute_threshold=0.01,
        )

        # Current is significantly worse
        baseline = self._create_results([0.9, 0.85, 0.88, 0.92, 0.87])
        current = self._create_results([0.6, 0.55, 0.58, 0.62, 0.57])

        report = detector.detect(current, baseline)
        assert report.has_regression
        assert report.score_delta < 0
        assert len(report.affected_tests) > 0

    def test_classify_severity(self):
        detector = RegressionDetector()

        critical = RegressionReport(has_regression=True, effect_size=1.0, score_delta=-0.3)
        assert detector.classify_severity(critical) == "critical"

        major = RegressionReport(has_regression=True, effect_size=0.6, score_delta=-0.08)
        assert detector.classify_severity(major) == "major"

        minor = RegressionReport(has_regression=True, effect_size=0.2, score_delta=-0.02)
        assert detector.classify_severity(minor) == "minor"

    def _create_results(self, scores: list[float]) -> list[TestResult]:
        results = []
        for i, score in enumerate(scores):
            tc = TestCase(id=f"tc_{i}", name=f"Test {i}")
            ao = AgentOutput(output=f"output {i}")
            results.append(TestResult(
                test_case=tc,
                agent_output=ao,
                overall_score=score,
                status=TestStatus.PASSED if score > 0.7 else TestStatus.FAILED,
            ))
        return results
