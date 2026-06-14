"""验证种子数据集可被框架正确加载并跑通评估。"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "areval-engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "areval-sdk"))

from areval.test_case import TestCase, AgentOutput, TestResult, EvaluationRun
from areval.datasets.formats import load_jsonl
from areval.metrics.accuracy import ExactMatchMetric, ContainsMetric
from areval.metrics.base import list_metrics
from areval.evaluator import Evaluator


def echo_agent(tc: TestCase) -> AgentOutput:
    """简单 echo agent：返回 expected_output（如果有），否则返回空。"""
    output = tc.expected_output or "No response available."
    return AgentOutput(
        output=output,
        tool_calls=[{"tool": t, "args": {}} for t in (tc.expected_tools or [])],
        latency_ms=50.0,
    )


def main():
    print("=" * 60)
    print("  AREval Seed Dataset Validation")
    print("=" * 60)

    datasets = {
        "customer_service": "datasets/seed/customer_service.jsonl",
        "rag_evaluation": "datasets/seed/rag_evaluation.jsonl",
        "safety_redteam": "datasets/seed/safety_redteam.jsonl",
    }

    all_pass = True

    for name, path in datasets.items():
        print(f"\n--- {name} ---")

        # 1. Load
        cases = load_jsonl(path)
        print(f"  Loaded: {len(cases)} cases")

        # 2. Basic validation
        errors = []
        for tc in cases:
            if not tc.name:
                errors.append(f"MISSING name in {tc.id}")
        if errors:
            for e in errors:
                print(f"  ERROR: {e}")
            all_pass = False
        else:
            print(f"  Validation: OK")

        # 3. Tag distribution
        tag_counts = Counter()
        for tc in cases:
            tag_counts.update(tc.tags)
        top_tags = tag_counts.most_common(5)
        print(f"  Top tags: {dict(top_tags)}")

        # 4. Quick evaluation with ExactMatchMetric
        evaluator = Evaluator(
            metrics=[ExactMatchMetric(case_sensitive=False)],
            threshold=0.5,
        )
        run = evaluator.evaluate(
            test_cases=cases[:5],  # Just test first 5
            agent_fn=echo_agent,
            run_name=f"validation-{name}",
            compare_baseline=False,
        )
        print(f"  Eval (first 5): pass_rate={run.pass_rate:.0%}, avg_score={run.avg_score:.2f}")

        # 5. Serialization round-trip
        run_dict = run.to_dict()
        json_str = json.dumps(run_dict, default=str)
        assert len(json_str) > 0, "Serialization failed"
        print(f"  Serialization: OK ({len(json_str)} bytes)")

    # Summary
    print(f"\n{'=' * 60}")
    if all_pass:
        print("  ALL VALIDATIONS PASSED")
    else:
        print("  SOME VALIDATIONS FAILED - see errors above")
    print(f"{'=' * 60}")

    # Print available metrics
    print(f"\nRegistered metrics: {list_metrics()}")


if __name__ == "__main__":
    main()
