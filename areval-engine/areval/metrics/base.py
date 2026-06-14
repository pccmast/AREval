"""Base metric classes with plugin architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from areval.test_case import TestCase, AgentOutput


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""

    name: str
    score: float  # 0.0 to 1.0
    passed: bool = False
    threshold: float = 0.7
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.passed = self.score >= self.threshold


class Metric(ABC):
    """Abstract base class for all evaluation metrics.

    The plugin architecture allows users to register custom metrics
    by subclassing Metric and implementing the measure() method.
    """

    name: str = "base_metric"
    threshold: float = 0.7

    def __init__(self, threshold: Optional[float] = None, **kwargs: Any):
        self.threshold = threshold or self.threshold
        self.config = kwargs

    @abstractmethod
    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        """Evaluate the agent output against the test case.

        Args:
            test_case: The test case with expected behavior
            agent_output: The actual output from the agent

        Returns:
            MetricResult with score and reasoning
        """
        pass

    def batch_measure(
        self, test_cases: List[TestCase], agent_outputs: List[AgentOutput]
    ) -> List[MetricResult]:
        """Evaluate multiple test cases in batch."""
        return [self.measure(tc, ao) for tc, ao in zip(test_cases, agent_outputs)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "config": self.config,
        }


# Metric registry for plugin system
_metric_registry: Dict[str, type] = {}


def register_metric(name: str, metric_class: type):
    """Register a custom metric class."""
    if not issubclass(metric_class, Metric):
        raise ValueError(f"Metric class must subclass Metric: {metric_class}")
    _metric_registry[name] = metric_class


def get_metric(name: str, **kwargs: Any) -> Metric:
    """Get a metric instance by name."""
    if name not in _metric_registry:
        raise KeyError(f"Unknown metric: {name}. Available: {list(_metric_registry.keys())}")
    return _metric_registry[name](**kwargs)


def list_metrics() -> List[str]:
    """List all registered metric names."""
    return list(_metric_registry.keys())
