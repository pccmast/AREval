"""Dataset format loaders.

Supports multiple formats including SWE-bench, JSONL, CSV,
and custom formats.
"""

import csv
import json
from typing import Dict, List, Optional

from areval.test_case import TestCase


def load_jsonl(path: str) -> List[TestCase]:
    """Load test cases from JSONL file."""
    test_cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            test_cases.append(TestCase.from_dict(data))
    return test_cases


def save_jsonl(test_cases: List[TestCase], path: str) -> None:
    """Save test cases to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for tc in test_cases:
            f.write(json.dumps(tc.to_dict(), default=str) + "\n")


def load_csv(path: str, mapping: Optional[Dict[str, str]] = None) -> List[TestCase]:
    """Load test cases from CSV file.

    Args:
        path: Path to CSV file
        mapping: Column mapping {test_case_field: csv_column}
    """
    default_mapping = {
        "name": "name",
        "input": "input",
        "expected_output": "expected_output",
        "context": "context",
        "tags": "tags",
    }
    mapping = mapping or default_mapping

    test_cases = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = {k: row.get(v, "") for k, v in mapping.items()}
            if "tags" in data and isinstance(data["tags"], str):
                data["tags"] = [t.strip() for t in data["tags"].split(",")]
            test_cases.append(TestCase.from_dict(data))
    return test_cases


def load_swe_bench(path: str, repo_filter: Optional[List[str]] = None) -> List[TestCase]:
    """Load SWE-bench format dataset.

    SWE-bench instances contain:
    - repo: Repository name
    - instance_id: Unique task ID
    - base_commit: Git commit hash
    - problem_statement: Issue description
    - patch: Reference solution
    - test_patch: Test cases to verify fix

    Args:
        path: Path to SWE-bench JSONL file
        repo_filter: Optional list of repo names to filter
    """
    test_cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            if repo_filter and data.get("repo") not in repo_filter:
                continue

            tc = TestCase(
                name=f"{data.get('repo', 'unknown')}-{data.get('instance_id', 'unknown')[:8]}",
                input=data.get("problem_statement", ""),
                expected_output=data.get("patch", ""),
                context=data.get("test_patch", ""),
                task_id=data.get("instance_id"),
                repository=data.get("repo"),
                base_commit=data.get("base_commit"),
                test_command="python -m pytest",  # Simplified
                tags=["swe-bench", data.get("repo", "unknown")],
            )
            test_cases.append(tc)
    return test_cases


def save_swe_bench_format(test_cases: List[TestCase], path: str) -> None:
    """Save test cases in SWE-bench compatible format."""
    records = []
    for tc in test_cases:
        record = {
            "instance_id": tc.task_id or tc.id,
            "repo": tc.repository or "unknown",
            "base_commit": tc.base_commit or "",
            "problem_statement": tc.input,
            "patch": tc.expected_output or "",
            "test_patch": tc.context or "",
        }
        records.append(record)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
