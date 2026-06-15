"""Baseline management for regression testing.

Manages golden sets, baseline versions, and historical comparison data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from areval.test_case import TestResult, EvaluationRun
from areval.utils.serialization import reconstruct_test_result


@dataclass
class Baseline:
    """A saved baseline for regression comparison."""

    id: str
    name: str
    description: str = ""
    run_id: str = ""
    test_results: List[TestResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "run_id": self.run_id,
            "test_results": [r.to_dict() for r in self.test_results],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }


class BaselineManager:
    """Manages baseline snapshots for regression testing.

    Supports:
    - Creating baselines from evaluation runs
    - Versioning baselines with tags
    - Loading historical baselines
    - Baseline drift detection
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".areval/baselines")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._baselines: Dict[str, Baseline] = {}
        self._load_all()

    def create_baseline(
        self,
        run: EvaluationRun,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Baseline:
        """Create a new baseline from an evaluation run."""
        baseline = Baseline(
            id=f"baseline-{run.id}",
            name=name or f"Baseline {run.name}",
            description=description,
            run_id=run.id,
            test_results=run.test_results,
            tags=tags or ["auto"],
        )
        self._baselines[baseline.id] = baseline
        self._save(baseline)
        return baseline

    def get_baseline(self, baseline_id: str) -> Optional[Baseline]:
        """Retrieve a baseline by ID."""
        return self._baselines.get(baseline_id)

    def get_latest_baseline(self, tag: Optional[str] = None) -> Optional[Baseline]:
        """Get the most recent baseline, optionally filtered by tag."""
        baselines = list(self._baselines.values())
        if tag:
            baselines = [b for b in baselines if tag in b.tags]

        if not baselines:
            return None

        return max(baselines, key=lambda b: b.created_at)

    def list_baselines(self) -> List[Baseline]:
        """List all saved baselines."""
        return sorted(self._baselines.values(), key=lambda b: b.created_at, reverse=True)

    def delete_baseline(self, baseline_id: str) -> bool:
        """Delete a baseline."""
        if baseline_id in self._baselines:
            del self._baselines[baseline_id]
            file_path = self.storage_path / f"{baseline_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def compare_to_baseline(
        self,
        results: List[TestResult],
        baseline_id: Optional[str] = None,
        delta_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Compare results against a baseline and return deltas.

        Parameters
        ----------
        results : list[TestResult]
            The current evaluation results to compare.
        baseline_id : str, optional
            ID of a specific baseline.  Defaults to the most recent.
        delta_threshold : float
            Minimum score drop to flag a test as regressed.
            Matches :attr:`RegressionDetector.absolute_threshold` (default 0.05).
        """
        baseline = self.get_baseline(baseline_id) if baseline_id else self.get_latest_baseline()
        if not baseline:
            return {"error": "No baseline found"}

        baseline_map = {r.test_case.id: r.overall_score for r in baseline.test_results}
        comparisons = []

        for result in results:
            baseline_score = baseline_map.get(result.test_case.id)
            if baseline_score is not None:
                comparisons.append({
                    "test_id": result.test_case.id,
                    "test_name": result.test_case.name,
                    "baseline_score": baseline_score,
                    "current_score": result.overall_score,
                    "delta": result.overall_score - baseline_score,
                    "regressed": result.overall_score < baseline_score - delta_threshold,
                })

        return {
            "baseline_id": baseline.id,
            "baseline_name": baseline.name,
            "comparisons": comparisons,
            "avg_delta": sum(c["delta"] for c in comparisons) / len(comparisons) if comparisons else 0,
            "regressed_count": sum(1 for c in comparisons if c["regressed"]),
        }

    def _save(self, baseline: Baseline) -> None:
        """Persist baseline to disk."""
        file_path = self.storage_path / f"{baseline.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(baseline.to_dict(), f, indent=2, default=str)

    def _load_all(self) -> None:
        """Load all baselines from storage including test results."""
        if not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                # Reconstruct test results
                test_results: list[TestResult] = []
                for tr_data in data.get("test_results", []):
                    test_results.append(reconstruct_test_result(tr_data))

                baseline = Baseline(
                    id=data["id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    run_id=data.get("run_id", ""),
                    test_results=test_results,
                    tags=data.get("tags", []),
                    metadata=data.get("metadata", {}),
                )
                self._baselines[baseline.id] = baseline
            except (json.JSONDecodeError, KeyError):
                continue
