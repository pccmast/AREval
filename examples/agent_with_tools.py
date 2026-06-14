"""Agent with tool evaluation example.

Demonstrates evaluating an agent that uses tools,
with tool call accuracy metrics and tracing.
"""

from areval.evaluator import Evaluator
from areval.test_case import TestCase, AgentOutput
from areval.metrics import ToolCallAccuracyMetric, TaskCompletionMetric
from areval.sdk.decorators import eval_trace
from areval.sdk.reporters import JSONReporter


# Simulate a tool-using agent with tracing
@eval_trace(name="weather_agent")
def weather_agent(query: str) -> tuple[str, list]:
    """An agent that fetches weather information."""
    tool_calls = []

    # Step 1: Extract location
    location = query.replace("weather in ", "").strip()
    tool_calls.append({"name": "extract_location", "params": {"query": query}})

    # Step 2: Fetch weather
    tool_calls.append({"name": "get_weather", "params": {"location": location}})

    # Step 3: Format response
    response = f"The weather in {location} is sunny and 72°F."
    tool_calls.append({"name": "format_response", "params": {"weather": "sunny"}})

    return response, tool_calls


def main():
    # Test cases with expected tool sequences
    test_cases = [
        TestCase(
            name="weather_query",
            input="What is the weather in San Francisco?",
            expected_output="The weather in San Francisco is sunny and 72°F.",
            expected_tools=["extract_location", "get_weather", "format_response"],
            tags=["tools", "weather"],
        ),
        TestCase(
            name="weather_nyc",
            input="weather in New York",
            expected_output="The weather in New York is sunny and 72°F.",
            expected_tools=["extract_location", "get_weather", "format_response"],
            tags=["tools", "weather"],
        ),
    ]

    evaluator = Evaluator(threshold=0.8)
    evaluator.add_metric(ToolCallAccuracyMetric(check_order=True, check_params=True))
    evaluator.add_metric(TaskCompletionMetric())

    def agent_fn(tc: TestCase) -> AgentOutput:
        output, tool_calls = weather_agent(tc.input)
        return AgentOutput(
            output=output,
            tool_calls=tool_calls,
            latency_ms=150.0,
            token_usage={"input": 25, "output": 15},
        )

    run = evaluator.evaluate(
        test_cases=test_cases,
        agent_fn=agent_fn,
        run_name="tool-agent-evaluation",
    )

    print(evaluator.summary(run))

    # Export for CI
    reporter = JSONReporter(run)
    reporter.export("tool_eval_results.json")
    print("\nResults exported to tool_eval_results.json")


if __name__ == "__main__":
    main()
