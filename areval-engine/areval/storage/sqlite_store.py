"""SQLite-backed :class:`~areval.storage.base.EvaluationStore`.

Uses SQLAlchemy ORM for persistence.  The schema stores the run metadata
in a relational table and the full ``test_results`` payload as a JSON
column (avoids a complex join-heavy schema while still benefiting from
indexed queries on aggregate fields).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from areval.storage.base import EvaluationStore
from areval.test_case import EvaluationRun
from areval.utils.serialization import reconstruct_run


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class _RunRow(_Base):
    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, default="")
    description = Column(String, default="")
    config_json = Column(Text, default="{}")
    payload_json = Column(Text, nullable=False)  # full run.to_dict()
    total_cases = Column(Integer, default=0)
    passed_cases = Column(Integer, default=0)
    failed_cases = Column(Integer, default=0)
    avg_score = Column(Float, default=0.0)
    pass_rate = Column(Float, default=0.0)
    total_cost_usd = Column(Float, default=0.0)
    regression_count = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SqliteStore(EvaluationStore):
    """SQLite-backed persistent store using SQLAlchemy ORM.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    echo : bool
        If True, log all SQL statements (useful for debugging).
    """

    def __init__(self, db_path: str | Path = ".areval/evaluations.db", echo: bool = False) -> None:
        self._engine = create_engine(f"sqlite:///{Path(db_path)}", echo=echo)
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, run: EvaluationRun) -> None:
        with self._Session() as session:
            row = _RunRow(
                id=run.id,
                name=run.name,
                description=run.description,
                config_json=json.dumps(run.config, default=str),
                payload_json=json.dumps(run.to_dict(), default=str),
                total_cases=run.total_cases,
                passed_cases=run.passed_cases,
                failed_cases=run.failed_cases,
                avg_score=run.avg_score,
                pass_rate=run.pass_rate,
                total_cost_usd=run.total_cost_usd,
                regression_count=run.regression_count,
                started_at=run.started_at,
                completed_at=run.completed_at,
            )
            session.merge(row)
            session.commit()

    def get(self, run_id: str) -> Optional[EvaluationRun]:
        with self._Session() as session:
            row = session.get(_RunRow, run_id)
            if row is None:
                return None
            return self._row_to_run(row)

    def list(self, limit: int = 100, offset: int = 0) -> List[EvaluationRun]:
        with self._Session() as session:
            rows = (
                session.query(_RunRow)
                .order_by(_RunRow.started_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._row_to_run(r) for r in rows]

    def delete(self, run_id: str) -> bool:
        with self._Session() as session:
            row = session.get(_RunRow, run_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def count(self) -> int:
        with self._Session() as session:
            return session.query(_RunRow).count()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_run(row: _RunRow) -> EvaluationRun:
        """Reconstruct an EvaluationRun from an ORM row."""
        data = json.loads(row.payload_json)
        return reconstruct_run(data)
