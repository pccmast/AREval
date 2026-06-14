"""Tests for evaluation metrics."""

import os

import pytest

from areval.test_case import TestCase, AgentOutput
from areval.metrics.accuracy import ExactMatchMetric, ContainsMetric, RegexMatchMetric
from areval.metrics.semantic import (
    SemanticSimilarityMetric,
    OfflineEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
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


# ------------------------------------------------------------------
# SemanticSimilarity tests (Sprint 1.2 / 1.5)
# ------------------------------------------------------------------


class TestOfflineEmbeddingProvider:
    """Tests for deterministic offline embeddings."""

    def test_same_text_same_vector(self) -> None:
        provider = OfflineEmbeddingProvider()
        v1 = provider.embed("hello world")
        v2 = provider.embed("hello world")
        import numpy as np
        assert np.allclose(v1, v2)
        assert v1.shape == (1536,)

    def test_different_text_different_vector(self) -> None:
        provider = OfflineEmbeddingProvider()
        v1 = provider.embed("hello")
        v2 = provider.embed("goodbye")
        import numpy as np
        assert not np.allclose(v1, v2)

    def test_cache_hit(self) -> None:
        provider = OfflineEmbeddingProvider()
        v1 = provider.embed("cache test")
        v2 = provider.embed("cache test")
        # Should be the same object (cached) or at least same values
        import numpy as np
        assert np.allclose(v1, v2)


class TestSemanticSimilarityMetric:
    """Tests for SemanticSimilarityMetric with offline provider."""

    def test_identical_output_similarity(self) -> None:
        metric = SemanticSimilarityMetric(embedding_provider="offline")
        tc = TestCase(name="t", input="hi", expected_output="hello world")
        ao = AgentOutput(output="hello world")

        result = metric.measure(tc, ao)
        assert 0.0 <= result.score <= 1.0
        # Identical text should have near-max similarity (offline hash-based
        # embeddings may have tiny floating-point differences)
        assert result.score > 0.99

    def test_no_expected_output(self) -> None:
        metric = SemanticSimilarityMetric()
        tc = TestCase(name="t", input="hi")
        ao = AgentOutput(output="anything")

        result = metric.measure(tc, ao)
        assert result.score == 0.0
        assert "No expected_output" in result.reasoning

    def test_openai_falls_back_to_offline(self) -> None:
        """When OPENAI_API_KEY is missing, openai provider falls back."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            metric = SemanticSimilarityMetric(embedding_provider="openai")
            tc = TestCase(name="t", input="hi", expected_output="hello")
            ao = AgentOutput(output="hello")

            result = metric.measure(tc, ao)
            assert 0.0 <= result.score <= 1.0
            assert result.passed == (result.score >= metric.threshold)
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_score_between_zero_and_one(self) -> None:
        metric = SemanticSimilarityMetric(embedding_provider="offline")
        tc = TestCase(name="t", input="hi", expected_output="hello world")
        ao = AgentOutput(output="something completely different")

        result = metric.measure(tc, ao)
        assert 0.0 <= result.score <= 1.0

    def test_metadata_includes_provider(self) -> None:
        metric = SemanticSimilarityMetric(embedding_provider="offline")
        tc = TestCase(name="t", input="hi", expected_output="hello")
        ao = AgentOutput(output="hello")

        result = metric.measure(tc, ao)
        assert "embedding_provider" in result.metadata
        assert "cosine_similarity" in result.metadata
