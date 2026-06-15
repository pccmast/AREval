"""Tests for upgraded agent metrics (Sprint 4).

Covers:
- ToolCallAccuracy with semantic matching (Tier 2)
- TaskCompletion modes: deterministic / open_ended / trajectory
- Backward compatibility
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areval.metrics.agent import (
    ToolCallAccuracyMetric,
    TaskCompletionMetric,
    _parse_semantic_batch,
)
from areval.test_case import TestCase, AgentOutput


def _make_tc(expected_tools: list[str] | None = None) -> TestCase:
    return TestCase(
        name="t",
        input="do something",
        expected_tools=expected_tools,
    )


def _make_ao(
    output: str = "",
    tool_calls: list[dict] | None = None,
    metadata: dict | None = None,
) -> AgentOutput:
    return AgentOutput(
        output=output,
        tool_calls=tool_calls or [],
        metadata=metadata or {},
    )


# ============================================================================
# Helper
# ============================================================================

class TestParseSemanticBatch:
    def test_all_yes(self):
        raw = "1: YES\n2: YES"
        result = _parse_semantic_batch(raw, 2)
        assert result == [True, True]

    def test_mixed(self):
        raw = "1: YES\n2: NO"
        result = _parse_semantic_batch(raw, 2)
        assert result == [True, False]

    def test_pads_to_count(self):
        raw = "1: YES"
        result = _parse_semantic_batch(raw, 3)
        assert result == [True, False, False]

    def test_alternate_formats(self):
        assert _parse_semantic_batch("1. YES\n2) NO", 2) == [True, False]


# ============================================================================
# ToolCallAccuracy — backward compat (semantic disabled)
# ============================================================================

class TestToolCallBackwardCompat:
    def test_correct_tools(self):
        m = ToolCallAccuracyMetric()
        tc = _make_tc(expected_tools=["get_order", "check_inventory"])
        ao = _make_ao(tool_calls=[
            {"name": "get_order"}, {"name": "check_inventory"},
        ])
        r = m.measure(tc, ao)
        assert r.score == 1.0

    def test_wrong_tools(self):
        m = ToolCallAccuracyMetric()
        tc = _make_tc(expected_tools=["get_order"])
        ao = _make_ao(tool_calls=[{"name": "wrong_tool"}])
        r = m.measure(tc, ao)
        assert r.score < 0.5

    def test_check_semantic_defaults_to_false(self):
        m = ToolCallAccuracyMetric()
        assert m.check_semantic is False


# ============================================================================
# ToolCallAccuracy — semantic matching
# ============================================================================

class TestToolCallSemantic:
    def test_semantic_match_uses_tier2(self):
        m = ToolCallAccuracyMetric(check_semantic=True)
        tc = _make_tc(expected_tools=["get_order", "query_inventory"])
        ao = _make_ao(tool_calls=[
            {"name": "fetch_order"}, {"name": "check_inventory"},
        ])

        mock_p = MagicMock()
        mock_p.is_available.return_value = True
        mock_p.chat_complete.return_value = "1: YES\n2: YES"

        with patch("areval.providers.local_llm.LocalLLMProvider", return_value=mock_p):
            r = m.measure(tc, ao)
            assert r.metadata.get("tier") == "tier2"
            assert r.score == 1.0

    def test_semantic_mixed_match(self):
        m = ToolCallAccuracyMetric(check_semantic=True)
        tc = _make_tc(expected_tools=["get_order", "delete_user"])
        ao = _make_ao(tool_calls=[
            {"name": "fetch_order"}, {"name": "create_user"},
        ])

        mock_p = MagicMock()
        mock_p.is_available.return_value = True
        mock_p.chat_complete.return_value = "1: YES\n2: NO"

        with patch("areval.providers.local_llm.LocalLLMProvider", return_value=mock_p):
            r = m.measure(tc, ao)
            assert r.metadata.get("tier") == "tier2"
            # 1/2 semantic matches + 1/2 params → 0.25 + 0.25 = 0.5
            assert r.score == pytest.approx(0.5)

    def test_semantic_silent_fallback_when_unavailable(self):
        """When local LLM unavailable, fall back to exact-name matching."""
        m = ToolCallAccuracyMetric(check_semantic=True)
        tc = _make_tc(expected_tools=["get_order"])
        ao = _make_ao(tool_calls=[{"name": "get_order"}])

        mock_p = MagicMock()
        mock_p.is_available.return_value = False

        with patch("areval.providers.local_llm.LocalLLMProvider", return_value=mock_p):
            r = m.measure(tc, ao)
            assert r.metadata.get("tier") == "tier1"
            assert r.score == 1.0

    def test_no_tools_returns_one(self):
        m = ToolCallAccuracyMetric(check_semantic=True)
        tc = _make_tc(expected_tools=[])
        ao = _make_ao()
        r = m.measure(tc, ao)
        assert r.score == 1.0


# ============================================================================
# TaskCompletionMetric
# ============================================================================

class TestTaskCompletion:
    def test_deterministic_mode(self):
        m = TaskCompletionMetric(mode="deterministic")
        tc = TestCase(name="t", input="anything")
        ao = _make_ao(output="This is a test output.")
        r = m.measure(tc, ao)
        assert r.metadata.get("mode") == "deterministic"
        assert r.metadata.get("tier") == "tier1"
        assert 0.0 <= r.score <= 1.0

    def test_deterministic_empty_output(self):
        m = TaskCompletionMetric(mode="deterministic")
        tc = TestCase(name="t", input="anything")
        ao = _make_ao(output="")
        r = m.measure(tc, ao)
        assert r.score == 0.0

    def test_deterministic_with_test_command(self):
        m = TaskCompletionMetric(mode="deterministic")
        tc = TestCase(name="t", input="test", test_command="pytest")
        ao = _make_ao(output="All tests passed successfully")
        r = m.measure(tc, ao)
        assert r.score == 1.0

    def test_open_ended_mode(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        m = TaskCompletionMetric(mode="open_ended")
        tc = TestCase(name="t", input="write code", expected_output="def foo(): pass")
        ao = _make_ao(output="def foo():\n    pass")

        from areval.metrics.base import MetricResult as MR
        fake = MR(
            name="task_completion", score=1.0, reasoning="complete",
            threshold=1.0, metadata={"tier": "tier3"},
        )
        with patch.object(m, "_evaluate_with_llm", return_value=fake):
            r = m.measure(tc, ao)
            assert r.metadata.get("tier") == "tier3"
            assert r.score == 1.0

    def test_trajectory_mode(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        m = TaskCompletionMetric(mode="trajectory")
        tc = TestCase(name="t", input="analyze data")
        ao = _make_ao(
            output="analysis complete",
            metadata={"trajectory": "step1: query; step2: analyze"},
        )

        from areval.metrics.base import MetricResult as MR
        fake = MR(
            name="task_completion", score=0.9, reasoning="good trajectory",
            threshold=1.0, metadata={"tier": "tier3", "mode": "trajectory"},
        )
        with patch.object(m, "_evaluate_with_llm", return_value=fake):
            r = m.measure(tc, ao)
            assert r.metadata.get("tier") == "tier3"
            assert r.metadata.get("mode") == "trajectory"

    def test_open_ended_falls_back_to_deterministic(self):
        """Without API key, open_ended falls to deterministic."""
        m = TaskCompletionMetric(mode="open_ended")
        tc = TestCase(name="t", input="test")
        ao = _make_ao(output="result")
        r = m.measure(tc, ao)
        assert r.metadata.get("tier") == "tier1"
        assert r.metadata.get("mode") == "open_ended"

    def test_mock_uses_deterministic(self):
        m = TaskCompletionMetric(provider="mock", mode="open_ended")
        tc = TestCase(name="t", input="test")
        ao = _make_ao(output="result")
        r = m.measure(tc, ao)
        assert r.metadata.get("tier") == "tier1"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            TaskCompletionMetric(mode="invalid")

    def test_llm_provider_raises_when_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        m = TaskCompletionMetric(provider="llm", mode="open_ended")
        with pytest.raises(RuntimeError, match="no OPENAI_API_KEY"):
            m.measure(TestCase(name="t", input="x"), _make_ao(output="y"))
