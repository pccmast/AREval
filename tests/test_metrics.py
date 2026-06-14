"""Tests for evaluation metrics."""

import pytest

from areval.test_case import TestCase, AgentOutput
from areval.metrics.accuracy import ExactMatchMetric, ContainsMetric, RegexMatchMetric
from areval.metrics.semantic import SemanticSimilarityMetric
from areval.metrics.rag import FaithfulnessMetric, AnswerRelevanceMetric
from areval.metrics.agent import ToolCallAccuracyMetric, TaskCompletionMetric


class TestExactMatchMetric:
    def test_perfect_match(self):
        metric = ExactMatchMetric()
        tc = TestCase(input="hello", expected_output="world")
        ao = AgentOutput(output="world")
        result = metric.measure(tc, ao)
        assert result.score == 1.0
        assert result.passed

    def test_mismatch(self):
        metric = ExactMatchMetric()
        tc = TestCase(input="hello", expected_output="world")
        ao = AgentOutput(output="earth")
        result = metric.measure(tc, ao)
        assert result.score == 0.0
        assert not result.passed

    def test_case_insensitive(self):
        metric = ExactMatchMetric(case_sensitive=False)
        tc = TestCase(expected_output="Hello")
        ao = AgentOutput(output="hello")
        result = metric.measure(tc, ao)
        assert result.score == 1.0


class TestContainsMetric:
    def test_contains_all(self):
        metric = ContainsMetric(all_required=True)
        tc = TestCase(expected_output="foo|bar")
        ao = AgentOutput(output="foo and bar")
        result = metric.measure(tc, ao)
        assert result.score == 1.0

    def test_contains_partial(self):
        metric = ContainsMetric(all_required=True)
        tc = TestCase(expected_output="foo|bar|baz")
        ao = AgentOutput(output="foo only")
        result = metric.measure(tc, ao)
        assert 0 < result.score < 1.0


class TestFaithfulnessMetric:
    def test_faithful_output(self):
        metric = FaithfulnessMetric()
        tc = TestCase(
            input="What is Python?",
            context="Python is a programming language created by Guido van Rossum.",
        )
        ao = AgentOutput(output="Python is a programming language.")
        result = metric.measure(tc, ao)
        assert result.score > 0.5

    def test_no_context(self):
        metric = FaithfulnessMetric()
        tc = TestCase(input="What is Python?")
        ao = AgentOutput(output="Python is a programming language.")
        result = metric.measure(tc, ao)
        assert result.score == 0.0


class TestToolCallAccuracyMetric:
    def test_correct_tools(self):
        metric = ToolCallAccuracyMetric()
        tc = TestCase(expected_tools=["search", "calculate"])
        ao = AgentOutput(
            tool_calls=[
                {"name": "search", "params": {"query": "test"}},
                {"name": "calculate", "params": {"expr": "1+1"}},
            ]
        )
        result = metric.measure(tc, ao)
        assert result.score > 0.5

    def test_wrong_tools(self):
        metric = ToolCallAccuracyMetric()
        tc = TestCase(expected_tools=["search"])
        ao = AgentOutput(tool_calls=[{"name": "calculate"}])
        result = metric.measure(tc, ao)
        assert result.score < 0.5
