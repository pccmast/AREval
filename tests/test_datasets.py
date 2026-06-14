"""Tests for Dataset management (Sprint 1.5)."""

import json
import tempfile
from pathlib import Path

import pytest

from areval.test_case import TestCase
from areval.datasets.manager import Dataset, DatasetManager


class TestDataset:
    """Tests for the Dataset data class."""

    def test_filter_by_tag(self) -> None:
        ds = Dataset(
            name="mixed",
            test_cases=[
                TestCase(name="a", tags=["math"]),
                TestCase(name="b", tags=["geo"]),
                TestCase(name="c", tags=["math", "geo"]),
            ],
        )
        filtered = ds.filter_by_tag("math")
        assert filtered.size == 2
        assert all("math" in tc.tags for tc in filtered.test_cases)

    def test_split(self) -> None:
        cases = [TestCase(name=f"tc-{i}") for i in range(10)]
        ds = Dataset(name="full", test_cases=cases)

        train, test = ds.split(train_ratio=0.8)
        assert train.size == 8
        assert test.size == 2
        assert "train" in train.tags
        assert "test" in test.tags
        assert train.parent_version == ds.id

    def test_size_property(self) -> None:
        ds = Dataset(
            name="sized",
            test_cases=[TestCase(name="a"), TestCase(name="b")],
        )
        assert ds.size == 2


class TestDatasetManager:
    """Tests for the DatasetManager lifecycle."""

    @pytest.fixture
    def tmp_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    def test_create_from_jsonl(self, tmp_dir: Path) -> None:
        # Write a small JSONL file
        jsonl_path = tmp_dir / "cases.jsonl"
        lines = [
            {"input": "What is 2+2?", "expected_output": "4"},
            {"input": "Capital of France?", "expected_output": "Paris"},
        ]
        jsonl_path.write_text("\n".join(json.dumps(line) for line in lines))

        dm = DatasetManager(storage_path=tmp_dir / ".areval/datasets")
        ds = dm.create_from_file(str(jsonl_path), name="mini-demo", format="jsonl")

        assert ds.name == "mini-demo"
        assert ds.size == 2
        assert isinstance(ds.test_cases[0], TestCase)

    def test_load_jsonl_file_not_found(self) -> None:
        dm = DatasetManager()
        with pytest.raises(FileNotFoundError):
            dm.create_from_file("nonexistent_file.jsonl", name="bad")

    def test_create_from_list_and_get(self, tmp_dir: Path) -> None:
        dm = DatasetManager(storage_path=tmp_dir / ".areval/datasets")
        cases = [
            TestCase(name="q1", input="hello", expected_output="world"),
        ]
        ds = dm.create_from_list(cases, name="my-list")

        retrieved = dm.get_dataset(ds.id)
        assert retrieved is not None
        assert retrieved.size == 1

    def test_list_datasets(self, tmp_dir: Path) -> None:
        dm = DatasetManager(storage_path=tmp_dir / ".areval/datasets")
        dm.create_from_list([], name="one")
        dm.create_from_list([], name="two")

        all_ds = dm.list_datasets()
        assert len(all_ds) == 2
        names = {d.name for d in all_ds}
        assert names == {"one", "two"}

    def test_get_dataset_missing(self) -> None:
        dm = DatasetManager()
        assert dm.get_dataset("no-such-id") is None

    def test_unsupported_format(self) -> None:
        dm = DatasetManager()
        with pytest.raises(ValueError, match="Unsupported format"):
            dm.create_from_file("dummy.txt", name="bad", format="xml")

    def test_save_and_reload(self, tmp_dir: Path) -> None:
        storage = tmp_dir / ".areval/datasets"
        dm1 = DatasetManager(storage_path=storage)
        ds = dm1.create_from_list(
            [TestCase(name="saved", input="hi")], name="persistent"
        )

        # New manager loading from same storage
        dm2 = DatasetManager(storage_path=storage)
        reloaded = dm2.get_dataset(ds.id)
        assert reloaded is not None
        assert reloaded.name == "persistent"
        assert reloaded.size == 1

    def test_dataset_to_dict_roundtrip(self) -> None:
        ds = Dataset(
            name="roundtrip",
            test_cases=[TestCase(name="t1", input="q")],
            tags=["test"],
        )
        data = ds.to_dict()
        assert data["name"] == "roundtrip"
        assert len(data["test_cases"]) == 1
