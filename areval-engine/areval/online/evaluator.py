"""Online evaluator — real-time scoring of every Agent call.

Evaluates ``TestCase`` + ``AgentOutput`` pairs as they arrive,
writes results to ``TimeSeriesStorage``, and delegates alerting
to ``QualityMonitor``.

Mode
----
- **sync**  (async_mode=False): ``evaluate()`` blocks until scoring
  is done and returns an ``OnlineResult``.
- **async** (async_mode=True): ``evaluate()`` queues the work in a
  background thread and returns ``None`` immediately.  The result
  is written to storage when the worker processes it.

Back-pressure
-------------
When ``async_mode`` is on, the internal queue is bounded by
``max_queue_size``.  If the queue is full, new tasks are dropped
and a ``ResourceWarning`` is emitted so operators can adjust
capacity.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from areval.metrics.base import Metric
from areval.judges.base import Judge
from areval.online.storage import TimeSeriesStorage, OnlineResult
from areval.online.monitors import QualityMonitor
from areval.test_case import TestCase, AgentOutput

logger = logging.getLogger(__name__)


class OnlineEvaluator:
    """Real-time evaluation engine.

    Parameters
    ----------
    metrics : list of Metric
    judges : list of Judge
    threshold : float
        Pass / fail threshold for ``overall_score``.
    storage : TimeSeriesStorage, optional
    monitor : QualityMonitor, optional
    async_mode : bool
        When True, scoring runs in a daemon thread.
    max_queue_size : int
        Maximum pending tasks (only relevant in async_mode).
    """

    def __init__(
        self,
        metrics: Optional[List[Metric]] = None,
        judges: Optional[List[Judge]] = None,
        threshold: float = 0.7,
        storage: Optional[TimeSeriesStorage] = None,
        monitor: Optional[QualityMonitor] = None,
        async_mode: bool = True,
        max_queue_size: int = 1000,
    ) -> None:
        self.metrics = metrics or []
        self.judges = judges or []
        self.threshold = threshold
        self.storage = storage or TimeSeriesStorage()
        self.monitor = monitor or QualityMonitor(storage=self.storage)
        self.async_mode = async_mode
        self.max_queue_size = max_queue_size

        # Async machinery
        self._queue: queue.Queue[Optional[tuple[TestCase, AgentOutput]]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        if async_mode:
            self._start_worker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
    ) -> Optional[OnlineResult]:
        """Evaluate one Agent call.

        Returns ``OnlineResult`` in sync mode; ``None`` in async mode.
        """
        if self.async_mode:
            try:
                self._queue.put_nowait((test_case, agent_output))
            except queue.Full:
                warnings.warn(
                    "OnlineEvaluator queue full — dropping task. "
                    f"Increase max_queue_size (currently {self.max_queue_size}).",
                    ResourceWarning,
                    stacklevel=2,
                )
            return None
        else:
            return self._evaluate_sync(test_case, agent_output)

    def evaluate_batch(
        self,
        test_cases: List[TestCase],
        agent_outputs: List[AgentOutput],
    ) -> List[OnlineResult]:
        """Batch evaluate (sync — useful for replay / backfill)."""
        results: List[OnlineResult] = []
        for tc, ao in zip(test_cases, agent_outputs):
            r = self._evaluate_sync(tc, ao)
            if r:
                results.append(r)
        return results

    def get_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        return self.storage.get_stats(window_minutes=window_minutes)

    def get_trend(
        self, window_minutes: int = 1440, bucket_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        return self.storage.get_trend(
            window_minutes=window_minutes,
            bucket_minutes=bucket_minutes,
            threshold=self.threshold,
        )

    def get_health(self) -> Dict[str, Any]:
        return self.monitor.get_health_status()

    def shutdown(self) -> None:
        """Graceful shutdown — wait for the async queue to drain."""
        if not self.async_mode or not self._worker_thread:
            return
        self._running = False
        self._queue.put(None)  # sentinel
        self._worker_thread.join(timeout=10.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            tc, ao = item
            try:
                result = self._evaluate_sync(tc, ao)
                if result:
                    self.storage.append(result)
                    self.monitor.check()
            except Exception:
                logger.exception(
                    "OnlineEvaluator worker: unhandled error evaluating task " "(test_case=%r)",
                    getattr(tc, "name", "unknown"),
                )
            finally:
                self._queue.task_done()

    def _evaluate_sync(self, test_case: TestCase, agent_output: AgentOutput) -> OnlineResult:
        scores: Dict[str, float] = {}
        input_hash = hashlib.md5(test_case.input.encode()).hexdigest()[:12]

        # Metrics
        for metric in self.metrics:
            try:
                r = metric.measure(test_case, agent_output)
                scores[r.name] = r.score
            except Exception:
                scores[metric.name] = 0.0

        # Judges
        for judge in self.judges:
            try:
                judge_result = judge.evaluate(test_case, agent_output)
                scores[judge.name] = judge_result.score
            except Exception:
                scores[judge.name] = 0.0

        overall = sum(scores.values()) / len(scores) if scores else 0.0

        return OnlineResult(
            timestamp=datetime.now(timezone.utc),
            trace_id=agent_output.trace_id,
            input_hash=input_hash,
            scores=scores,
            overall_score=overall,
            passed=overall >= self.threshold,
            latency_ms=agent_output.latency_ms,
            cost_usd=agent_output.cost_usd,
            metadata={"test_case_name": test_case.name},
        )
