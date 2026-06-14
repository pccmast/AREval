"""Evaluation metrics for AREval.

Built-in metrics inspired by DeepEval, Ragas, and SWE-bench patterns.
"""

from areval.metrics.base import Metric, MetricResult
from areval.metrics.accuracy import ExactMatchMetric, ContainsMetric, RegexMatchMetric
from areval.metrics.semantic import SemanticSimilarityMetric
from areval.metrics.rag import FaithfulnessMetric, AnswerRelevanceMetric, ContextPrecisionMetric
from areval.metrics.agent import ToolCallAccuracyMetric, TaskCompletionMetric

__all__ = [
    "Metric",
    "MetricResult",
    "ExactMatchMetric",
    "ContainsMetric",
    "RegexMatchMetric",
    "SemanticSimilarityMetric",
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "ToolCallAccuracyMetric",
    "TaskCompletionMetric",
]
