"""Storage abstraction layer.

Provides a pluggable backend interface for persisting evaluation runs
so that the business logic (API, CLI, Evaluator) is decoupled from the
concrete storage engine.

Available backends
------------------
* :class:`JsonFileStore` – JSON file on disk (current behaviour)
* :class:`SqliteStore`   – SQLite via SQLAlchemy ORM

Future backends (PostgreSQL, Redis, S3, …) implement
:class:`EvaluationStore` without touching any business code.
"""

from areval.storage.base import EvaluationStore
from areval.storage.json_store import JsonFileStore
from areval.storage.sqlite_store import SqliteStore

__all__ = ["EvaluationStore", "JsonFileStore", "SqliteStore"]
