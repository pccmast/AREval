"""Time-series storage for online evaluation results.

Uses JSONL (append-only lines) for the MVP.  Production deployments
should swap this implementation for PostgreSQL + TimescaleDB without
changing the public interface.
"""

from __future__ import annotations

import json
import threading
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OnlineResult:
    """A single online evaluation result."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: Optional[str] = None
    input_hash: str = ""        # hash of the user input (for dedup / correlation)
    scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    passed: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnlineResult":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            data["timestamp"] = datetime.fromisoformat(ts)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TimeSeriesStorage:
    """JSONL-backed time-series store for online evaluation results.

    Thread-safe for concurrent writes.

    .. note::
       Maximum records is enforced to prevent unbounded disk usage.
       Oldest records are pruned when the limit is exceeded.
    """

    DEFAULT_MAX_RECORDS = 100_000

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> None:
        self.storage_path = storage_path or Path(".areval/online")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self._file = self.storage_path / "results.jsonl"
        self._lock = threading.Lock()
        self._prune_on_next_write = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, result: OnlineResult) -> None:
        """Append a single result (thread-safe)."""
        with self._lock:
            # Prune if over limit (best-effort: truncation of entire file)
            self._try_prune()
            with open(self._file, "a") as f:
                f.write(json.dumps(result.to_dict(), default=str) + "\n")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        min_score: Optional[float] = None,
        passed_only: Optional[bool] = None,
    ) -> List[OnlineResult]:
        """Query results within a time range and optional filters."""
        results = self._load_all()
        if start:
            results = [r for r in results if r.timestamp >= start]
        if end:
            results = [r for r in results if r.timestamp <= end]
        if min_score is not None:
            results = [r for r in results if r.overall_score >= min_score]
        if passed_only is True:
            results = [r for r in results if r.passed]
        return results

    def get_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Aggregate stats for the last *window_minutes*."""
        cutoff = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = cutoff - timedelta(minutes=window_minutes)
        recent = self.query(start=start)

        total = len(recent)
        passed = sum(1 for r in recent if r.passed)
        avg_score = sum(r.overall_score for r in recent) / total if total else 0.0
        avg_latency = sum(r.latency_ms for r in recent) / total if total else 0.0

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_score": avg_score,
            "avg_latency_ms": avg_latency,
        }

    def get_trend(
        self,
        window_minutes: int = 1440,
        bucket_minutes: int = 60,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Return bucketed time-series trend data.

        Parameters
        ----------
        window_minutes : int
            Total time window to cover.
        bucket_minutes : int
            Size of each bucket (must be > 0).
        threshold : float
            Pass/fail threshold used to compute ``pass_rate`` per bucket.
        """
        cutoff = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = cutoff - timedelta(minutes=window_minutes)
        results = self.query(start=start)

        # Build empty buckets
        buckets: Dict[str, List[float]] = {}
        t = start
        while t <= cutoff:
            buckets[t.isoformat()] = []
            t += timedelta(minutes=bucket_minutes)
            if bucket_minutes <= 0:
                break

        # Fill buckets
        for r in results:
            ts = r.timestamp.replace(second=0, microsecond=0)
            # Find the bucket start
            bucket_start = ts - timedelta(
                minutes=ts.minute % bucket_minutes
            )
            key = bucket_start.isoformat()
            if key in buckets:
                buckets[key].append(r.overall_score)

        return [
            {
                "timestamp": ts,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "pass_rate": sum(1 for s in scores if s >= threshold) / len(scores)
                if scores
                else 0.0,
                "count": len(scores),
            }
            for ts, scores in buckets.items()
        ]

    def clear(self, before: Optional[datetime] = None) -> int:
        """Delete records; returns count of deleted entries."""
        with self._lock:
            if before is None:
                count = sum(1 for _ in self._load_all_gen())
                self._file.write_text("")
                return count

            kept: List[str] = []
            deleted = 0
            for line in self._load_raw_lines():
                try:
                    r = OnlineResult.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue
                if r.timestamp < before:
                    deleted += 1
                else:
                    kept.append(line)
            self._file.write_text("".join(kept))
            return deleted

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_prune(self) -> None:
        """If the file exceeds max_records, truncate oldest entries."""
        count = sum(1 for _ in self._load_raw_lines())
        if count <= self.max_records:
            return
        excess = count - self.max_records
        warnings.warn(
            f"TimeSeriesStorage: pruning {excess} oldest records (max={self.max_records})",
            stacklevel=3,
        )
        kept = list(self._load_raw_lines())[excess:]
        self._file.write_text("".join(kept))

    def _load_raw_lines(self):
        if not self._file.exists():
            return
        with open(self._file) as f:
            for line in f:
                if line.strip():
                    yield line

    def _load_all_gen(self):
        for line in self._load_raw_lines():
            try:
                yield OnlineResult.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue

    def _load_all(self) -> List[OnlineResult]:
        return list(self._load_all_gen())
