"""Dataset management with versioning and curation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from areval.test_case import TestCase
from areval.datasets.formats import load_jsonl, save_jsonl


@dataclass
class Dataset:
    """A versioned collection of test cases."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    test_cases: List[TestCase] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: int = 1
    parent_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.test_cases)

    def filter_by_tag(self, tag: str) -> Dataset:
        """Return a new dataset with only test cases matching the tag."""
        filtered = [tc for tc in self.test_cases if tag in tc.tags]
        return Dataset(
            name=f"{self.name}-{tag}",
            test_cases=filtered,
            tags=self.tags + [tag],
            parent_version=self.id,
        )

    def split(self, train_ratio: float = 0.8) -> tuple[Dataset, Dataset]:
        """Split dataset into train and test sets."""
        split_idx = int(len(self.test_cases) * train_ratio)
        train_cases = self.test_cases[:split_idx]
        test_cases = self.test_cases[split_idx:]

        train = Dataset(
            name=f"{self.name}-train",
            test_cases=train_cases,
            tags=self.tags + ["train"],
            parent_version=self.id,
        )
        test = Dataset(
            name=f"{self.name}-test",
            test_cases=test_cases,
            tags=self.tags + ["test"],
            parent_version=self.id,
        )
        return train, test

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "tags": self.tags,
            "version": self.version,
            "parent_version": self.parent_version,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class DatasetManager:
    """Manages dataset lifecycle: creation, versioning, curation.

    Supports:
    - Creating datasets from files (JSONL, CSV, SWE-bench)
    - Dataset versioning with lineage tracking
    - Train/test splits
    - Dataset curation and filtering
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".areval/datasets")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._datasets: Dict[str, Dataset] = {}
        self._load_all()

    def create_from_file(
        self,
        path: str,
        name: str,
        description: str = "",
        format: str = "jsonl",
    ) -> Dataset:
        """Create a dataset from a file."""
        if format == "jsonl":
            test_cases = load_jsonl(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        dataset = Dataset(
            name=name,
            description=description,
            test_cases=test_cases,
            tags=[format],
        )
        self._datasets[dataset.id] = dataset
        self._save(dataset)
        return dataset

    def create_from_list(
        self,
        test_cases: List[TestCase],
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dataset:
        """Create a dataset from a list of test cases."""
        dataset = Dataset(
            name=name,
            description=description,
            test_cases=test_cases,
            tags=tags or [],
        )
        self._datasets[dataset.id] = dataset
        self._save(dataset)
        return dataset

    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Retrieve a dataset by ID."""
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> List[Dataset]:
        """List all datasets."""
        return sorted(self._datasets.values(), key=lambda d: d.created_at, reverse=True)

    def save_dataset(self, dataset: Dataset) -> None:
        """Save a dataset to storage."""
        self._datasets[dataset.id] = dataset
        self._save(dataset)

    def _save(self, dataset: Dataset) -> None:
        """Persist dataset to disk."""
        file_path = self.storage_path / f"{dataset.id}.json"
        with open(file_path, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2, default=str)

    def _load_all(self) -> None:
        """Load all datasets from storage."""
        if not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                test_cases = [TestCase.from_dict(tc) for tc in data.get("test_cases", [])]
                dataset = Dataset(
                    id=data["id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    test_cases=test_cases,
                    tags=data.get("tags", []),
                    version=data.get("version", 1),
                    parent_version=data.get("parent_version"),
                    metadata=data.get("metadata", {}),
                )
                self._datasets[dataset.id] = dataset
            except (json.JSONDecodeError, KeyError):
                continue
