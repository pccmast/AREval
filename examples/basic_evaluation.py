"""Basic evaluation example.

Demonstrates how to use AREval to evaluate a simple agent function.
"""

from areval.evaluator import Evaluator
from areval.test_case import TestCase, AgentOutput
from areval.metrics import ExactMatchMetric, SemanticSimilarityMetric
from areval.judges import LLMJudge


def my_simple_agent(input_text: str) -> str:
    """A simple agent for demonstration."""
    responses = {
        "What is 2+2?": "4",
        "Capital of France?": "Paris",
        "Hello": "Hello! How can I help you?",
    }
    return responses.get(input_text, "I don't know.")


def main():
    # 1. Define test cases
    test_cases = [
        TestCase(
            name="math_basic",
            input="What is 2+2?",
            expected_output="4",
            tags=["math"],
        ),
        TestCase(
            name="geography",
            input="Capital of France?",
            expected_output="Paris",
            tags=["geography"],
        ),
        TestCase(
            name="greeting",
            input="Hello",
            expected_output="Hello! How can I help you?",
            tags=["conversation"],
        ),
    ]

    # 2. Build evaluator with metrics and judges
    evaluator = Evaluator(threshold=0.7)
    evaluator.add_metric(ExactMatchMetric())
    evaluator.add_metric(SemanticSimilarityMetric())
    evaluator.add_judge(LLMJudge(threshold=0.6))

    # 3. Run evaluation
    def agent_fn(tc: TestCase) -> AgentOutput:
        output = my_simple_agent(tc.input)
        return AgentOutput(output=output, latency_ms=50.0)

    run = evaluator.evaluate(
        test_cases=test_cases,
        agent_fn=agent_fn,
        run_name="basic-demo",
        run_description="Simple agent evaluation",
    )

    # 4. Print results
    print(evaluator.summary(run))

    # 5. Save as baseline
    baseline_id = evaluator.create_baseline(run, name="v1-baseline")
    print(f"\nBaseline saved: {baseline_id}")


if __name__ == "__main__":
    main()
