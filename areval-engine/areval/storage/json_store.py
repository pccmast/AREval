"""JSON-file backed :class:`~areval.storage.base.EvaluationStore`.

This is the default backend, mirroring the current production behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from areval.storage.base import EvaluationStore
from areval.test_case import EvaluationRun
from areval.utils.serialization import reconstruct_run


class JsonFileStore(EvaluationStore):
    """Store evaluation runs as individual JSON files on disk.

    Each run is saved as ``<run_id>.json`` under the configured directory.
    An in-memory cache avoids re-reading the disk on every access.
    """

    def __init__(self, directory: str | Path = ".areval/runs") -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, EvaluationRun] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, run: EvaluationRun) -> None:
        self._cache[run.id] = run
        file_path = self._dir / f"{run.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

    def get(self, run_id: str) -> Optional[EvaluationRun]:
        return self._cache.get(run_id)

    def list(self, limit: int = 100, offset: int = 0) -> List[EvaluationRun]:
        runs = sorted(
            self._cache.values(),
            key=lambda r: r.started_at,
            reverse=True,
        )
        return runs[offset : offset + limit]

    def delete(self, run_id: str) -> bool:
        if run_id not in self._cache:
            return False
        del self._cache[run_id]
        file_path = self._dir / f"{run_id}.json"
        if file_path.exists():
            file_path.unlink()
        return True

    def count(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all persisted runs from disk on startup."""
        if not self._dir.exists():
            return
        for file_path in self._dir.glob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                run = reconstruct_run(data)
                self._cache[run.id] = run
            except (json.JSONDecodeError, KeyError):
                continue
