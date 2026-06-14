"""SDK decorators for agent evaluation integration.

Usage:
    @eval_trace(name="my_agent")
    def my_agent(input_text: str) -> str:
        # Agent logic
        return response

    @eval_metric(metric="exact_match", threshold=1.0)
    def test_my_agent():
        return my_agent("hello")
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional

from areval.test_case import AgentOutput
from areval.tracing.tracer import EvalTracer

# Global tracer instance
_tracer = EvalTracer()


def eval_trace(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable[..., Any]:
    """Decorator to trace agent function execution.

    Captures latency, input/output, and custom attributes.
    Integrates with the evaluation tracing system.

    Example:
        @eval_trace(name="search_agent")
        def search(query: str) -> str:
            return search_api(query)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or func.__name__
            span_attrs = attributes or {}

            if capture_input:
                span_attrs["input.args"] = str(args)[:500]
                span_attrs["input.kwargs"] = str(kwargs)[:500]

            with _tracer.start_span(span_name, attributes=span_attrs) as span:
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    if capture_output:
                        span.set_attribute("output", str(result)[:500])
                    span.set_attribute("latency_ms", (time.time() - start) * 1000)
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        return wrapper
    return decorator


def eval_metric(
    metric: str = "exact_match",
    threshold: float = 0.7,
    expected: Optional[str] = None,
) -> Callable:
    """Decorator to evaluate a function's output against a metric.

    Example:
        @eval_metric(metric="contains", threshold=1.0, expected="answer")
        def get_answer(question: str) -> str:
            return llm.generate(question)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Quick inline metric check
            from areval.metrics.accuracy import ContainsMetric, ExactMatchMetric

            metric_map = {
                "exact_match": ExactMatchMetric(threshold=threshold),
                "contains": ContainsMetric(threshold=threshold),
            }

            if expected and metric in metric_map:
                from areval.test_case import TestCase
                m = metric_map[metric]
                tc = TestCase(input=str(args), expected_output=expected)
                ao = AgentOutput(output=str(result))
                metric_result = m.measure(tc, ao)

                if not metric_result.passed:
                    print(f"⚠ Metric check failed: {metric_result.reasoning}")

            return result
        return wrapper
    return decorator


def get_tracer() -> EvalTracer:
    """Get the global tracer instance."""
    return _tracer
