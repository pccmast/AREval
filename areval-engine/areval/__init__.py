"""AREval - Agent Regression Evaluation Harness.

The core evaluation engine for benchmarking and regression testing AI agents.
"""

__version__ = "0.1.0"

from areval.evaluator import Evaluator
from areval.test_case import TestCase, TestResult
from areval.metrics.base import Metric, MetricResult
from areval.judges.base import Judge, JudgeResult
from areval.regression.detector import RegressionDetector

__all__ = [
    "Evaluator",
    "TestCase",
    "TestResult",
    "Metric",
    "MetricResult",
    "Judge",
    "JudgeResult",
    "RegressionDetector",
]
