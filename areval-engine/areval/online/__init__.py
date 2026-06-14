"""Online evaluation module — real-time quality monitoring.

Evaluates every Agent call as it happens (not just batch tests),
writes results to time-series storage, and triggers alerts on
quality degradation.
"""

from areval.online.storage import TimeSeriesStorage, OnlineResult
from areval.online.monitors import QualityMonitor, AlertConfig, Alert
from areval.online.evaluator import OnlineEvaluator

__all__ = [
    "OnlineEvaluator",
    "TimeSeriesStorage",
    "OnlineResult",
    "QualityMonitor",
    "AlertConfig",
    "Alert",
]
