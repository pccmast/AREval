"""Base judge classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from areval.test_case import TestCase, AgentOutput


@dataclass
class JudgeResult:
    """Result from a judge evaluation."""

    score: float  # 0.0 to 1.0
    reasoning: str = ""
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    threshold: float = 0.7

    def __post_init__(self):
        self.passed = self.score >= self.threshold


class Judge(ABC):
    """Abstract base class for judges.

    Judges use LLM or agentic reasoning to evaluate outputs
    with nuanced, human-like assessment.
    """

    name: str = "base_judge"
    threshold: float = 0.7

    def __init__(self, threshold: Optional[float] = None, **kwargs: Any):
        self.threshold = threshold or self.threshold
        self.config = kwargs

    @abstractmethod
    def evaluate(self, test_case: TestCase, agent_output: AgentOutput) -> JudgeResult:
        """Evaluate an agent output and return a scored judgement."""
        pass

    def batch_evaluate(
        self, test_cases: List[TestCase], agent_outputs: List[AgentOutput]
    ) -> List[JudgeResult]:
        """Evaluate multiple test cases."""
        return [self.evaluate(tc, ao) for tc, ao in zip(test_cases, agent_outputs)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "config": self.config,
        }


# Judge registry
_judge_registry: Dict[str, type] = {}


def register_judge(name: str, judge_class: type):
    """Register a custom judge class."""
    if not issubclass(judge_class, Judge):
        raise ValueError(f"Judge class must subclass Judge: {judge_class}")
    _judge_registry[name] = judge_class


def get_judge(name: str, **kwargs: Any) -> Judge:
    """Get a judge instance by name."""
    if name not in _judge_registry:
        raise KeyError(f"Unknown judge: {name}. Available: {list(_judge_registry.keys())}")
    return _judge_registry[name](**kwargs)
