"""Basic evaluation example.

Demonstrates: TestCase → Agent → Evaluator(metrics + judges) → Summary.

Run with:
    uv run python examples/basic_evaluation.py
"""

import os
import sys
from pathlib import Path

# -- 自动加载 .env（可选，本地模型 token 等）--
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from areval.evaluator import Evaluator
from areval.test_case import TestCase, AgentOutput
from areval.metrics import ExactMatchMetric, SemanticSimilarityMetric
from areval.judges import LLMJudge


def my_agent(input_text: str) -> str:
    responses = {
        "What is 2+2?": "4",
        "Capital of France?": "Paris",
        "Hello": "Hello! How can I help you?",
    }
    return responses.get(input_text, "I don't know.")


def main():
    # ---- Test cases ----
    test_cases = [
        TestCase(name="math",    input="What is 2+2?",       expected_output="4",                        tags=["math"]),
        TestCase(name="geo",     input="Capital of France?", expected_output="Paris",                     tags=["geo"]),
        TestCase(name="greet",   input="Hello",              expected_output="Hello! How can I help you?", tags=["conv"]),
    ]
    print("Test cases:")
    for tc in test_cases:
        print(f"  {tc.name}: '{tc.input}' -> expected '{tc.expected_output}'")

    # ---- Evaluator (offline -- no API key needed) ----
    evaluator = (
        Evaluator(threshold=0.7)
        .add_metric(ExactMatchMetric(case_sensitive=False))
        .add_metric(SemanticSimilarityMetric(embedding_provider="offline"))
        .add_judge(LLMJudge(provider="mock"))
    )

    def agent_fn(tc: TestCase) -> AgentOutput:
        output = my_agent(tc.input)
        print(f"  Agent '{tc.name}': {tc.input!r} -> {output!r}")
        return AgentOutput(output=output, latency_ms=50.0)

    print("\nRunning evaluation...")
    run = evaluator.evaluate(test_cases=test_cases, agent_fn=agent_fn,
                             run_name="basic-demo", run_description="3 cases, offline mode")

    # ---- Per-case detail ----
    print("\nPer-case results:")
    for tr in run.test_results:
        status = "PASS" if tr.passed else "FAIL"
        print(f"  [{status}] {tr.test_case.name}: score={tr.overall_score:.3f} "
              f"(exact={tr.scores.get('exact_match',0):.1f}, "
              f"sem={tr.scores.get('semantic_similarity',0):.1f}, "
              f"judge={tr.scores.get('llm_judge',0):.1f})")

    # ---- Summary ----
    print("\n" + evaluator.summary(run))


if __name__ == "__main__":
    main()
