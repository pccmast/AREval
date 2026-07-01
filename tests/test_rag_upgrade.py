"""Tests for upgraded RAG metrics (Sprint 2).

Covers:
- Three-tier fallback behaviour for all three RAG metrics
- Provider dispatch: mock / local / llm / auto
- Faithfulness sentence splitting, batch prompt, and scoring
- ContextPrecision / AnswerRelevance binary classification
- Long-text auto-upgrade (Faithfulness)
- Helper functions (normalise label, split sentences, parse batch)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areval.metrics.rag import (
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    _split_sentences,
    _normalise_label,
    _normalise_binary,
    _parse_faithfulness_batch,
    _compute_faithfulness_score,
)
from areval.test_case import TestCase, AgentOutput


# ============================================================================
# Helpers
# ============================================================================

def _make_test_case(
    input_text: str = "What is ML?",
    context: str = "ML is a field of AI.",
    expected: str = "ML stands for machine learning.",
) -> TestCase:
    return TestCase(input=input_text, context=context, expected_output=expected)


def _make_agent_output(output: str = "ML is machine learning.") -> AgentOutput:
    return AgentOutput(output=output)


def _mock_local_provider(reply: str) -> MagicMock:
    """Create a mock LocalLLMProvider that returns *reply*."""
    mock = MagicMock()
    mock.is_available.return_value = True
    mock.chat_complete.return_value = reply
    return mock


# ============================================================================
# Pure-function tests (no external deps)
# ============================================================================

class TestSentenceSplit:
    def test_empty(self):
        assert _split_sentences("") == []

    def test_single(self):
        assert _split_sentences("Hello world.") == ["Hello world"]

    def test_multiple(self):
        result = _split_sentences("Hello. World! How are you?")
        assert result == ["Hello", "World", "How are you"]

    def test_trailing_no_sentence(self):
        result = _split_sentences("Hello. World")
        assert result == ["Hello", "World"]


class TestNormaliseLabel:
    def test_exact_match(self):
        assert _normalise_label("SUPPORTED", ("SUPPORTED", "NEUTRAL"), "NEUTRAL") == "SUPPORTED"

    def test_case_insensitive(self):
        assert _normalise_label("supported  ", ("SUPPORTED",), "NEUTRAL") == "SUPPORTED"

    def test_prefix_match(self):
        assert _normalise_label("SUPPORTED: yes", ("SUPPORTED", "CONTRADICTED"), "NEUTRAL") == "SUPPORTED"

    def test_contains_match(self):
        assert _normalise_label("the answer is SUPPORTED by context", ("SUPPORTED",), "NEUTRAL") == "SUPPORTED"

    def test_fallback_to_default(self):
        assert _normalise_label("garbage", ("RELEVANT", "NOT_RELEVANT"), "NOT_RELEVANT") == "NOT_RELEVANT"


class TestNormaliseBinary:
    def test_relevant(self):
        assert _normalise_binary("RELEVANT") == 1.0
        assert _normalise_binary("  relevant\n") == 1.0

    def test_not_relevant(self):
        assert _normalise_binary("NOT_RELEVANT") == 0.0
        assert _normalise_binary("garbage") == 0.0


class TestParseFaithfulnessBatch:
    def test_parses_correctly(self):
        raw = "1: SUPPORTED\n2: CONTRADICTED\n3: NEUTRAL"
        assert _parse_faithfulness_batch(raw, 3) == ["SUPPORTED", "CONTRADICTED", "NEUTRAL"]

    def test_pads_missing(self):
        raw = "1: SUPPORTED"
        result = _parse_faithfulness_batch(raw, 3)
        assert result == ["SUPPORTED", "NEUTRAL", "NEUTRAL"]

    def test_truncates_extra(self):
        raw = "1: SUPPORTED\n2: SUPPORTED\n3: SUPPORTED\n4: CONTRADICTED"
        result = _parse_faithfulness_batch(raw, 2)
        assert len(result) == 2

    def test_alternate_formats(self):
        assert _parse_faithfulness_batch("1 SUPPORTED\n2. CONTRADICTED", 2) == ["SUPPORTED", "CONTRADICTED"]
        assert _parse_faithfulness_batch("1) SUPPORTED\n2) NEUTRAL", 2) == ["SUPPORTED", "NEUTRAL"]


class TestComputeFaithfulnessScore:
    def test_all_supported(self):
        results = [("s1", "SUPPORTED"), ("s2", "SUPPORTED")]
        assert _compute_faithfulness_score(results) == 1.0

    def test_all_contradicted(self):
        results = [("s1", "CONTRADICTED"), ("s2", "CONTRADICTED")]
        assert _compute_faithfulness_score(results) == 0.0

    def test_neutral_default_zero(self):
        results = [("s1", "NEUTRAL"), ("s2", "NEUTRAL")]
        assert _compute_faithfulness_score(results, neutral_weight=0.0) == 0.0

    def test_neutral_weighted(self):
        results = [("s1", "NEUTRAL"), ("s2", "SUPPORTED")]
        assert _compute_faithfulness_score(results, neutral_weight=0.5) == 0.75

    def test_empty_results(self):
        assert _compute_faithfulness_score([]) == 1.0

    def test_mixed(self):
        results = [
            ("s1", "SUPPORTED"),
            ("s2", "SUPPORTED"),
            ("s3", "CONTRADICTED"),
            ("s4", "NEUTRAL"),
        ]
        # SUPPORTED=2*1 + CONTRADICTED=1*0 + NEUTRAL=1*0 = 2 / 4 = 0.5
        assert _compute_faithfulness_score(results, neutral_weight=0.0) == 0.5


# ============================================================================
# Metric dispatch tests (all use mocks — CI-safe)
# ============================================================================

class TestFaithfulnessMetric:
    def test_mock_provider_uses_tier1(self):
        """provider='mock' → straight to Tier 1 (LLMJudge mock)."""
        m = FaithfulnessMetric(provider="mock")
        tc = _make_test_case(context="ML is a branch of AI.")
        ao = _make_agent_output("ML means machine learning.")
        result = m.measure(tc, ao)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata.get("tier") == "tier1"

    def test_auto_falls_back_to_tier1_when_no_llm(self):
        """When neither local nor remote is available → Tier 1 mock."""
        m = FaithfulnessMetric(provider="auto")
        tc = _make_test_case(context="ML is a branch of AI.")
        ao = _make_agent_output("ML means machine learning.")

        with patch("areval.routing.tier2_available", return_value=False):
            result = m.measure(tc, ao)
            assert 0.0 <= result.score <= 1.0
            assert result.metadata.get("tier") == "tier1"

    def test_tier2_path_called_when_available(self):
        """When local LLM is reachable → Tier 2 is used."""
        m = FaithfulnessMetric(provider="auto")
        tc = _make_test_case(context="Paris is the capital of France.")
        ao = _make_agent_output("Paris is in France. It is a beautiful city.")

        mock_provider = _mock_local_provider(
            "1: SUPPORTED\n2: SUPPORTED\n"
        )

        with patch("areval.routing.tier2_available", return_value=True):
            with patch(
                "areval.providers.local_llm.LocalLLMProvider",
                return_value=mock_provider,
            ):
                result = m.measure(tc, ao)
                assert result.metadata.get("tier") == "tier2"
                # Both sentences supported → score = 1.0
                assert result.score == 1.0

    def test_faithfulness_tier2_contradicted(self):
        """Contradicted sentences → score drops."""
        m = FaithfulnessMetric(provider="auto")
        tc = _make_test_case(context="Paris is in France.")
        ao = _make_agent_output("Paris is in Germany. It is nice.")

        mock_provider = _mock_local_provider(
            "1: CONTRADICTED\n2: SUPPORTED\n"
        )

        with patch("areval.routing.tier2_available", return_value=True):
            with patch(
                "areval.providers.local_llm.LocalLLMProvider",
                return_value=mock_provider,
            ):
                result = m.measure(tc, ao)
                assert result.metadata.get("tier") == "tier2"
                # 1 SUPPORTED, 1 CONTRADICTED → 0.5
                assert result.score == 0.5

    def test_no_context_returns_zero(self):
        m = FaithfulnessMetric(provider="auto")
        tc = TestCase(input="Q", context="", expected_output="")
        ao = _make_agent_output("answer")
        result = m.measure(tc, ao)
        assert result.score == 0.0
        assert "No context" in (result.reasoning or "")

    def test_local_provider_raises_when_unavailable(self):
        m = FaithfulnessMetric(provider="local")
        tc = _make_test_case()
        ao = _make_agent_output()
        with patch("areval.routing.tier2_available", return_value=False):
            with pytest.raises(RuntimeError, match="local LLM is not available"):
                m.measure(tc, ao)

    def test_llm_provider_raises_when_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        m = FaithfulnessMetric(provider="llm")
        tc = _make_test_case()
        ao = _make_agent_output()
        with pytest.raises(RuntimeError, match="no OPENAI_API_KEY"):
            m.measure(tc, ao)

    def test_long_text_auto_upgrades_to_tier3(self, monkeypatch):
        """context+answer > complexity_threshold → Tier 3 if API key available."""
        from areval.metrics.base import MetricResult as MR

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        m = FaithfulnessMetric(provider="auto", complexity_threshold=50)

        tc = _make_test_case(context="x" * 60)  # context is 60 chars > 50
        ao = _make_agent_output("y")

        fake_t3_result = MR(
            name="faithfulness", score=0.95,
            reasoning="tier3 mock", threshold=0.7,
            metadata={"tier": "tier3"},
        )

        with patch("areval.routing.tier2_available", return_value=True):
            with patch.object(m, "_evaluate_tier3", return_value=fake_t3_result):
                result = m.measure(tc, ao)
                assert result.metadata.get("tier") == "tier3"
                assert result.score == 0.95


class TestAnswerRelevanceMetric:
    def test_mock_provider_uses_tier1(self):
        m = AnswerRelevanceMetric(provider="mock")
        tc = _make_test_case()
        ao = _make_agent_output("ML is machine learning.")
        result = m.measure(tc, ao)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata.get("tier") == "tier1"

    def test_auto_falls_back_to_tier1_when_no_llm(self):
        m = AnswerRelevanceMetric(provider="auto")
        tc = _make_test_case()
        ao = _make_agent_output("answer")

        with patch("areval.routing.tier2_available", return_value=False):
            result = m.measure(tc, ao)
            assert result.metadata.get("tier") == "tier1"

    def test_tier2_relevant(self):
        m = AnswerRelevanceMetric(provider="auto")
        tc = _make_test_case(input_text="What is ML?")
        ao = _make_agent_output("ML is machine learning.")

        mock_provider = _mock_local_provider("RELEVANT")

        with patch("areval.routing.tier2_available", return_value=True):
            with patch(
                "areval.providers.local_llm.LocalLLMProvider",
                return_value=mock_provider,
            ):
                result = m.measure(tc, ao)
                assert result.metadata.get("tier") == "tier2"
                assert result.score == 1.0

    def test_tier2_not_relevant(self):
        m = AnswerRelevanceMetric(provider="auto")
        tc = _make_test_case(input_text="What is ML?")
        ao = _make_agent_output("I like pizza.")

        mock_provider = _mock_local_provider("NOT_RELEVANT")

        with patch("areval.routing.tier2_available", return_value=True):
            with patch(
                "areval.providers.local_llm.LocalLLMProvider",
                return_value=mock_provider,
            ):
                result = m.measure(tc, ao)
                assert result.metadata.get("tier") == "tier2"
                assert result.score == 0.0

    def test_empty_question_vacuously_relevant(self):
        m = AnswerRelevanceMetric(provider="auto")
        tc = _make_test_case(input_text="")
        ao = _make_agent_output("anything")
        result = m.measure(tc, ao)
        assert result.score == 1.0

    def test_local_provider_raises_when_unavailable(self):
        m = AnswerRelevanceMetric(provider="local")
        tc = _make_test_case()
        ao = _make_agent_output()
        with patch("areval.routing.tier2_available", return_value=False):
            with pytest.raises(RuntimeError, match="local LLM is not available"):
                m.measure(tc, ao)


class TestContextPrecisionMetric:
    def test_mock_provider_uses_tier1(self):
        m = ContextPrecisionMetric(provider="mock")
        tc = _make_test_case(context="ML is a branch of AI.")
        ao = _make_agent_output()
        result = m.measure(tc, ao)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata.get("tier") == "tier1"

    def test_auto_falls_back_to_tier1_when_no_llm(self):
        m = ContextPrecisionMetric(provider="auto")
        tc = _make_test_case(context="ML is AI.")
        ao = _make_agent_output()

        with patch("areval.routing.tier2_available", return_value=False):
            result = m.measure(tc, ao)
            assert result.metadata.get("tier") == "tier1"

    def test_tier2_relevant(self):
        m = ContextPrecisionMetric(provider="auto")
        tc = _make_test_case(
            input_text="What is ML?",
            context="ML stands for machine learning.",
        )
        ao = _make_agent_output()

        mock_provider = _mock_local_provider("RELEVANT")

        with patch("areval.routing.tier2_available", return_value=True):
            with patch(
                "areval.providers.local_llm.LocalLLMProvider",
                return_value=mock_provider,
            ):
                result = m.measure(tc, ao)
                assert result.metadata.get("tier") == "tier2"
                assert result.score == 1.0

    def test_tier2_not_relevant(self):
        m = ContextPrecisionMetric(provider="auto")
        tc = _make_test_case(
            input_text="What is ML?",
            context="The weather is nice today.",
        )
        ao = _make_agent_output()

        mock_provider = _mock_local_provider("NOT_RELEVANT")

        with patch("areval.routing.tier2_available", return_value=True):
            with patch(
                "areval.providers.local_llm.LocalLLMProvider",
                return_value=mock_provider,
            ):
                result = m.measure(tc, ao)
                assert result.metadata.get("tier") == "tier2"
                assert result.score == 0.0

    def test_no_context_returns_zero(self):
        m = ContextPrecisionMetric(provider="auto")
        tc = TestCase(input="Q", context="", expected_output="")
        ao = _make_agent_output()
        result = m.measure(tc, ao)
        assert result.score == 0.0

    def test_local_provider_raises_when_unavailable(self):
        m = ContextPrecisionMetric(provider="local")
        tc = _make_test_case()
        ao = _make_agent_output()
        with patch("areval.routing.tier2_available", return_value=False):
            with pytest.raises(RuntimeError, match="local LLM is not available"):
                m.measure(tc, ao)


# ============================================================================
# Backward-compatibility: existing test patterns still work
# ============================================================================

class TestBackwardCompatibility:
    """Ensure existing metrics.py tests still pass with the new impl."""

    def test_faithfulness_has_correct_name(self):
        m = FaithfulnessMetric()
        assert m.name == "faithfulness"

    def test_answer_relevance_has_correct_name(self):
        m = AnswerRelevanceMetric()
        assert m.name == "answer_relevance"

    def test_context_precision_has_correct_name(self):
        m = ContextPrecisionMetric()
        assert m.name == "context_precision"

    def test_to_dict_includes_provider(self):
        m = FaithfulnessMetric(provider="auto")
        d = m.to_dict()
        assert d["provider"] == "auto"
        assert d["name"] == "faithfulness"

    def test_provider_default_is_auto(self):
        m = FaithfulnessMetric()
        assert m.provider == "auto"
