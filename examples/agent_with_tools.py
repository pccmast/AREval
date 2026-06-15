"""Agent tool-call evaluation example.

Demonstrates: ToolCallAccuracy + TaskCompletion with @eval_trace.

Run with:
    uv run python examples/agent_with_tools.py
"""

import os
from pathlib import Path

# -- 自动加载 .env --
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from areval.evaluator import Evaluator
from areval.test_case import TestCase, AgentOutput
from areval.metrics import ToolCallAccuracyMetric, TaskCompletionMetric
from areval_sdk.decorators import eval_trace
from areval_sdk.reporters import JSONReporter


@eval_trace(name="weather_agent")
def weather_agent(query: str):
    """Simulated tool-using agent."""
    import re
    # Extract city name: everything after "in " (dropping trailing punctuation)
    m = re.search(r'\bin\s+(.+?)[?.,!]*$', query, re.IGNORECASE)
    location = m.group(1).strip() if m else query.strip()
    tool_calls = [
        {"name": "extract_location", "params": {"query": query}},
        {"name": "get_weather", "params": {"location": location}},
        {"name": "format_response", "params": {"weather": "sunny"}},
    ]
    response = f"The weather in {location} is sunny and 72F."
    return response, tool_calls


def main():
    # ---- Test cases (with expected tool chains) ----
    test_cases = [
        TestCase(
            name="sf_weather",
            input="What is the weather in San Francisco?",
            expected_output="The weather in San Francisco is sunny and 72F.",
            expected_tools=["extract_location", "get_weather", "format_response"],
            tags=["tools", "weather"],
        ),
        TestCase(
            name="nyc_weather",
            input="weather in New York",
            expected_output="The weather in New York is sunny and 72F.",
            expected_tools=["extract_location", "get_weather", "format_response"],
            tags=["tools", "weather"],
        ),
    ]

    print("Test cases:")
    for tc in test_cases:
        print(f"  {tc.name}: '{tc.input}'")
        print(f"    expected tools: {tc.expected_tools}")

    # ---- Evaluator ----
    evaluator = (
        Evaluator(threshold=0.6)  # lower threshold bc TaskCompletion heuristic is weak offline
        .add_metric(ToolCallAccuracyMetric())
        .add_metric(TaskCompletionMetric(threshold=0.2))
    )

    def agent_fn(tc: TestCase) -> AgentOutput:
        output, tool_calls = weather_agent(tc.input)
        print(f"\n  Agent '{tc.name}':")
        print(f"    input:  {tc.input!r}")
        print(f"    output: {output!r}")
        print(f"    tools:  {' -> '.join(c['name'] for c in tool_calls)}")
        return AgentOutput(output=output, tool_calls=tool_calls,
                           latency_ms=150.0, token_usage={"input": 25, "output": 15})

    print("\nRunning evaluation...")
    run = evaluator.evaluate(test_cases=test_cases, agent_fn=agent_fn,
                             run_name="tool-agent-evaluation")

    # ---- Per-case ----
    print("\nPer-case results:")
    for tr in run.test_results:
        status = "PASS" if tr.passed else "FAIL"
        print(f"  [{status}] {tr.test_case.name}: score={tr.overall_score:.3f} "
              f"(tool_acc={tr.scores.get('tool_call_accuracy',0):.1f}, "
              f"task={tr.scores.get('task_completion',0):.2f})")

    # ---- Summary + export ----
    print("\n" + evaluator.summary(run))
    JSONReporter(run).export("tool_eval_results.json")
    print("Exported → tool_eval_results.json")


if __name__ == "__main__":
    main()
