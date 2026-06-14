"""Abstract storage interface for evaluation runs.

All production storage backends implement this ABC so that the
business logic is never coupled to a specific engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from areval.test_case import EvaluationRun


class EvaluationStore(ABC):
    """Pluggable storage backend for :class:`~areval.test_case.EvaluationRun`.

    Implementation notes
    --------------------
    * ``save()`` is idempotent — writing the same run twice overwrites.
    * ``list()`` returns runs sorted by ``started_at`` descending.
    * ``delete()`` returns ``True`` if the run existed, ``False`` otherwise.
    """

    @abstractmethod
    def save(self, run: EvaluationRun) -> None:
        """Persist (or overwrite) an evaluation run."""
        ...

    @abstractmethod
    def get(self, run_id: str) -> Optional[EvaluationRun]:
        """Retrieve a single run by its ID."""
        ...

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[EvaluationRun]:
        """List runs, most recent first."""
        ...

    @abstractmethod
    def delete(self, run_id: str) -> bool:
        """Delete a run.  Returns ``True`` if it existed."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of stored runs."""
        ...
