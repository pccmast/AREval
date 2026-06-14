"""Tests for online evaluation module (Phase 3)."""

import time
from datetime import datetime, timezone, timedelta

from areval.online.storage import TimeSeriesStorage, OnlineResult
from areval.online.monitors import QualityMonitor, AlertConfig
from areval.online.evaluator import OnlineEvaluator
from areval.metrics.accuracy import ExactMatchMetric
from areval.test_case import TestCase, AgentOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(score: float, passed: bool = True) -> OnlineResult:
    return OnlineResult(
        timestamp=datetime.now(timezone.utc),
        overall_score=score,
        passed=passed,
        latency_ms=100.0,
        input_hash="abc",
    )


# ---------------------------------------------------------------------------
# TimeSeriesStorage
# ---------------------------------------------------------------------------


class TestStorage:
    def test_append_and_query(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        for _ in range(5):
            s.append(_result(0.9))
        assert len(s.query()) == 5

    def test_time_filter(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        old = datetime.now(timezone.utc) - timedelta(hours=3)
        r_old = OnlineResult(timestamp=old, overall_score=0.5, passed=False, input_hash="x")
        s.append(r_old)
        s.append(_result(0.8))

        start = datetime.now(timezone.utc) - timedelta(hours=1)
        recent = s.query(start=start)
        assert len(recent) == 1
        assert recent[0].overall_score == 0.8

    def test_get_stats(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        for _ in range(8):
            s.append(_result(0.8, True))
        for _ in range(2):
            s.append(_result(0.3, False))
        stats = s.get_stats(window_minutes=60)
        assert stats["total"] == 10
        assert stats["passed"] == 8
        assert stats["failed"] == 2

    def test_get_trend(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        for i in range(6):
            ts = datetime.now(timezone.utc) - timedelta(minutes=i * 20)
            s.append(OnlineResult(timestamp=ts, overall_score=0.5 + i * 0.05, passed=True, input_hash="t"))
        trend = s.get_trend(window_minutes=180, bucket_minutes=60)
        assert len(trend) >= 2

    def test_clear(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        for _ in range(3):
            s.append(_result(0.5))
        assert s.clear() == 3
        assert len(s.query()) == 0


# ---------------------------------------------------------------------------
# QualityMonitor
# ---------------------------------------------------------------------------


class TestMonitor:
    def test_no_alert_when_healthy(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        for _ in range(20):
            s.append(_result(0.9))
        m = QualityMonitor(s, config=AlertConfig(pass_rate_threshold=0.7, window_minutes=60, min_samples=10))
        alerts = m.check()
        assert alerts == []

    def test_pass_rate_alert(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        for _ in range(10):
            s.append(_result(0.3, False))
        m = QualityMonitor(s, config=AlertConfig(pass_rate_threshold=0.7, window_minutes=60, min_samples=5))
        alerts = m.check()
        assert any(a.alert_type == "pass_rate_drop" for a in alerts)

    def test_cooldown(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        for _ in range(10):
            s.append(_result(0.3, False))
        m = QualityMonitor(s, config=AlertConfig(pass_rate_threshold=0.7, window_minutes=60, min_samples=5, cooldown_minutes=60))
        alerts1 = m.check()
        assert len(alerts1) > 0
        # Immediately again — should be suppressed
        alerts2 = m.check()
        assert alerts2 == []

    def test_min_samples(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        s.append(_result(0.1, False))
        s.append(_result(0.1, False))
        m = QualityMonitor(s, config=AlertConfig(min_samples=10, window_minutes=60))
        alerts = m.check()
        assert alerts == []

    def test_health_status_degraded(self) -> None:
        s = TimeSeriesStorage()
        s.clear()
        for _ in range(10):
            s.append(_result(0.5, False))
        m = QualityMonitor(s, config=AlertConfig(pass_rate_threshold=0.8, window_minutes=60, min_samples=2))
        health = m.get_health_status()
        assert health["status"] in ("degraded", "critical")


# ---------------------------------------------------------------------------
# OnlineEvaluator
# ---------------------------------------------------------------------------


class TestOnlineEvaluator:
    def test_sync_evaluate(self) -> None:
        e = OnlineEvaluator(
            metrics=[ExactMatchMetric()],
            async_mode=False,
        )
        e.storage.clear()
        tc = TestCase(name="t", input="hello", expected_output="hello")
        ao = AgentOutput(output="hello")
        r = e.evaluate(tc, ao)
        assert r is not None
        assert 0.0 <= r.overall_score <= 1.0
        assert "exact_match" in r.scores

    def test_sync_evaluate_score_zero_on_mismatch(self) -> None:
        e = OnlineEvaluator(
            metrics=[ExactMatchMetric()],
            async_mode=False,
        )
        e.storage.clear()
        tc = TestCase(name="t", input="hello", expected_output="hello")
        ao = AgentOutput(output="goodbye")
        r = e.evaluate(tc, ao)
        assert r is not None
        assert r.overall_score == 0.0
        assert not r.passed

    def test_async_evaluate(self) -> None:
        e = OnlineEvaluator(
            metrics=[ExactMatchMetric()],
            async_mode=True,
            max_queue_size=100,
        )
        e.storage.clear()
        tc = TestCase(name="t", input="hi", expected_output="hi")
        ao = AgentOutput(output="hi")
        result = e.evaluate(tc, ao)
        assert result is None  # async returns None immediately
        # Give worker time
        time.sleep(0.3)
        results = e.storage.query()
        assert len(results) >= 1

    def test_queue_overflow(self) -> None:
        e = OnlineEvaluator(
            metrics=[ExactMatchMetric()],
            async_mode=True,
            max_queue_size=3,
        )
        tc = TestCase(name="t", input="x", expected_output="x")
        ao = AgentOutput(output="x")
        # Fill queue
        for _ in range(10):
            e.evaluate(tc, ao)  # should emit ResourceWarning, not raise
        # Cleanup
        e.shutdown()

    def test_health(self) -> None:
        e = OnlineEvaluator(metrics=[ExactMatchMetric()], async_mode=False)
        health = e.get_health()
        assert "status" in health

    def test_shutdown(self) -> None:
        e = OnlineEvaluator(async_mode=True, max_queue_size=5)
        e.shutdown()
        # Should not raise
