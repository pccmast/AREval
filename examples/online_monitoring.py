"""Online evaluation & quality monitoring example.

Demonstrates: OnlineEvaluator + QualityMonitor with simulated traffic.

Run with:
    uv run python examples/online_monitoring.py
"""

import os
from pathlib import Path

# -- 自动加载 .env --
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from areval.online.evaluator import OnlineEvaluator
from areval.online.monitors import AlertConfig, QualityMonitor
from areval.online.storage import TimeSeriesStorage
from areval.metrics.accuracy import ExactMatchMetric
from areval.test_case import TestCase, AgentOutput


def main():
    storage = TimeSeriesStorage()
    monitor = QualityMonitor(
        storage=storage,
        config=AlertConfig(pass_rate_threshold=0.7, window_minutes=5, min_samples=3),
    )
    evaluator = OnlineEvaluator(
        metrics=[ExactMatchMetric(case_sensitive=False)],
        threshold=0.7, storage=storage, monitor=monitor,
        async_mode=False,
    )

    # Simulated traffic — expected vs actual
    print("Simulating 5 production requests:\n")
    traffic = [
        ("Capital of France?", "Paris",                        "Paris",                    True),
        ("What is 2+2?",       "4",                            "4",                        True),
        ("Explain Python",     "Python is a programming lang", "Python is a snake",        False),  # false info
        ("Quantum computing",  "Quantum uses qubits",          "",                         False),  # empty
        ("Weather today?",     "Sunny with 25C",               "Sunny with 25C",           True),
    ]

    for q, expected, actual, should_pass in traffic:
        tc = TestCase(name=f"live", input=q, expected_output=expected)
        ao = AgentOutput(output=actual, latency_ms=100)
        result = evaluator.evaluate(tc, ao)

        icon = "PASS" if (result and result.passed) else "FAIL"
        expt = "[OK]" if should_pass else "[XX]"
        print(f"  [{icon} {expt}] expected='{expected}' | actual='{actual}' | score={result.overall_score:.0f}" if result else f"  [{icon} {expt}] no result")

    # Stats
    # Health check
    health = evaluator.get_health()
    print(f"\nHealth: status={health['status']}, active_alerts={health['active_alerts']}")

    evaluator.shutdown()


if __name__ == "__main__":
    main()
