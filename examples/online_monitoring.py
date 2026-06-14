"""Online evaluation and quality monitoring example.

Demonstrates:
1. Creating an OnlineEvaluator with metrics and alerting
2. Simulating production traffic (some good, some degraded)
3. Checking real-time stats and health status
4. Getting trend data for dashboard display
"""

from areval.online.evaluator import OnlineEvaluator
from areval.online.monitors import AlertConfig, Alert, QualityMonitor
from areval.online.storage import TimeSeriesStorage
from areval.metrics.accuracy import ExactMatchMetric, ContainsMetric
from areval.metrics.semantic import SemanticSimilarityMetric
from areval.test_case import TestCase, AgentOutput


def on_alert(alert: Alert) -> None:
    print(f"  [ALERT {alert.severity.upper()}] {alert.message}")


def main() -> None:
    # -- Setup --
    storage = TimeSeriesStorage()
    monitor = QualityMonitor(
        storage=storage,
        config=AlertConfig(
            pass_rate_threshold=0.7,
            window_minutes=5,
            min_samples=3,
        ),
        alert_callback=on_alert,
    )
    evaluator = OnlineEvaluator(
        metrics=[ExactMatchMetric(), ContainsMetric(), SemanticSimilarityMetric()],
        threshold=0.7,
        storage=storage,
        monitor=monitor,
        async_mode=False,
    )

    # -- Simulate production traffic --
    print("Simulating production traffic...\n")
    scenarios = [
        ("What is the weather today?", "The weather is sunny and 25C.", "good"),
        ("Capital of France?", "Paris", "good"),
        ("Explain quantum computing.", "Quantum computing uses qubits that can...", "good"),
        ("What is Python?", "Python is a snake.", "bad"),          # false info
        ("Write a poem about AI.", "Roses are red, circuits are blue...", "good"),
        ("How to make pancakes?", "", "empty"),                    # empty output → bad
        ("Today's news", "I don't know the latest.", "ambiguous"),
        ("What is 2+2?", "4", "good"),
    ]

    for query, answer, label in scenarios:
        tc = TestCase(name=f"live-{label}", input=query, expected_output=answer)
        ao = AgentOutput(output=answer, latency_ms=100)
        result = evaluator.evaluate(tc, ao)
        if result:
            icon = "PASS" if result.passed else "FAIL"
            print(f"  [{icon}] {label:10s} | score={result.overall_score:.2f} | {query[:40]}...")

    # -- Stats --
    stats = evaluator.get_stats(window_minutes=5)
    print(f"\nStats (last 5 min): pass_rate={stats['pass_rate']:.1%}, avg_score={stats['avg_score']:.3f}")

    # -- Health --
    health = evaluator.get_health()
    print(f"Health: {health['status']} (pass_rate={health['pass_rate']:.1%}, alerts={health['active_alerts']})")

    # -- Trend --
    trend = evaluator.get_trend(window_minutes=60, bucket_minutes=5)
    print(f"\nTrend buckets: {len(trend)}")

    # Cleanup
    evaluator.shutdown()


if __name__ == "__main__":
    main()
