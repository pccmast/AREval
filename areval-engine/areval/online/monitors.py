"""Quality monitor — sliding-window detection and alerting.

Watches online evaluation results and fires alerts when quality
drops below configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

from areval.online.storage import TimeSeriesStorage


@dataclass
class AlertConfig:
    """Alerting configuration for :class:`QualityMonitor`.

    Parameters
    ----------
    pass_rate_threshold : float
        Trigger when pass_rate drops below this value.
    avg_score_threshold : float
        Trigger when average score drops below this value.
    window_minutes : int
        Sliding window size in minutes.
    min_samples : int
        Do NOT alert when sample count is below this (avoids
        false positives from low-traffic periods).
    cooldown_minutes : int
        Suppress repeat alerts of the same type within this window.
    """

    pass_rate_threshold: float = 0.7
    avg_score_threshold: float = 0.6
    window_minutes: int = 30
    min_samples: int = 10
    cooldown_minutes: int = 15


@dataclass
class Alert:
    """A single quality alert."""

    alert_type: str  # "pass_rate_drop" | "score_drop" | "latency_spike"
    severity: str  # "warning" | "critical"
    message: str
    current_value: float
    threshold_value: float
    window_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class QualityMonitor:
    """Sliding-window quality monitor with cooldown.

    Parameters
    ----------
    storage : TimeSeriesStorage
    config : AlertConfig, optional
    alert_callback : Callable[[Alert], None], optional
        Called for every triggered alert (e.g. send to Slack / log).
    """

    def __init__(
        self,
        storage: TimeSeriesStorage,
        config: Optional[AlertConfig] = None,
        alert_callback: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        self.storage = storage
        self.config = config or AlertConfig()
        self.alert_callback = alert_callback
        self._last_alert_times: Dict[str, datetime] = {}
        self._alert_history: List[Alert] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def check(self) -> List[Alert]:
        """Check current quality and return triggered alerts."""
        stats = self.storage.get_stats(window_minutes=self.config.window_minutes)
        alerts: List[Alert] = []

        if stats["total"] < self.config.min_samples:
            return alerts

        # Pass-rate alert
        if stats["pass_rate"] < self.config.pass_rate_threshold:
            a = self._maybe_alert(
                "pass_rate_drop",
                (
                    "critical"
                    if stats["pass_rate"] < self.config.pass_rate_threshold / 2
                    else "warning"
                ),
                f"Pass rate {stats['pass_rate']:.1%} < {self.config.pass_rate_threshold:.1%}",
                stats["pass_rate"],
                self.config.pass_rate_threshold,
                stats,
            )
            if a:
                alerts.append(a)

        # Score alert
        if stats["avg_score"] < self.config.avg_score_threshold:
            a = self._maybe_alert(
                "score_drop",
                (
                    "critical"
                    if stats["avg_score"] < self.config.avg_score_threshold / 2
                    else "warning"
                ),
                f"Avg score {stats['avg_score']:.3f} < {self.config.avg_score_threshold:.3f}",
                stats["avg_score"],
                self.config.avg_score_threshold,
                stats,
            )
            if a:
                alerts.append(a)

        return alerts

    def get_health_status(self) -> Dict[str, Any]:
        """Return a high-level health summary."""
        stats = self.storage.get_stats(window_minutes=self.config.window_minutes)
        alerts = self.check()

        if stats["total"] < self.config.min_samples:
            status = "unknown"
        elif stats["pass_rate"] < self.config.pass_rate_threshold / 2:
            status = "critical"
        elif stats["pass_rate"] < self.config.pass_rate_threshold:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "pass_rate": stats["pass_rate"],
            "avg_score": stats["avg_score"],
            "sample_count": stats["total"],
            "active_alerts": len(alerts),
        }

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent alert history for Dashboard / API consumption."""
        return [
            {
                "type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "current_value": a.current_value,
                "threshold_value": a.threshold_value,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in self._alert_history[-limit:]
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        current_value: float,
        threshold_value: float,
        stats: Dict[str, Any],
    ) -> Optional[Alert]:
        if self._is_in_cooldown(alert_type):
            return None
        self._last_alert_times[alert_type] = datetime.now(timezone.utc)

        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            window_stats=stats,
        )
        self._alert_history.append(alert)
        # Keep history bounded (most recent 500)
        if len(self._alert_history) > 500:
            self._alert_history = self._alert_history[-500:]
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception:
                pass
        return alert

    def _is_in_cooldown(self, alert_type: str) -> bool:
        last = self._last_alert_times.get(alert_type)
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last) < timedelta(minutes=self.config.cooldown_minutes)
