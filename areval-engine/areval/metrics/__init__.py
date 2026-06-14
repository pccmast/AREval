"""Evaluation metrics for AREval.

Built-in metrics inspired by DeepEval, Ragas, and SWE-bench patterns.

All built-in metrics are auto-registered at import time so that CLI YAML
configs can reference them by name via ``get_metric(name, **config)``.
"""

from areval.metrics.base import Metric, MetricResult, get_metric, list_metrics, register_metric
from areval.metrics.accuracy import ExactMatchMetric, ContainsMetric, RegexMatchMetric
from areval.metrics.semantic import SemanticSimilarityMetric
from areval.metrics.rag import FaithfulnessMetric, AnswerRelevanceMetric, ContextPrecisionMetric
from areval.metrics.agent import ToolCallAccuracyMetric, TaskCompletionMetric

# ---------------------------------------------------------------------------
# Auto-register built-in metrics
# ---------------------------------------------------------------------------
register_metric("exact_match", ExactMatchMetric)
register_metric("contains", ContainsMetric)
register_metric("regex_match", RegexMatchMetric)
register_metric("semantic_similarity", SemanticSimilarityMetric)
register_metric("faithfulness", FaithfulnessMetric)
register_metric("answer_relevance", AnswerRelevanceMetric)
register_metric("context_precision", ContextPrecisionMetric)
register_metric("tool_call_accuracy", ToolCallAccuracyMetric)
register_metric("task_completion", TaskCompletionMetric)

__all__ = [
    "Metric",
    "MetricResult",
    "get_metric",
    "list_metrics",
    "register_metric",
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
