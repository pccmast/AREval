"""Tests for trace collection and export.

Covers :class:`~areval.tracing.tracer.TraceSpan` and
:class:`~areval.tracing.tracer.EvalTracer`.
"""

import time

from areval.tracing.tracer import EvalTracer, TraceSpan


class TestTraceSpan:
    """Unit tests for TraceSpan."""

    def test_span_creation(self):
        span = TraceSpan(name="test_span")
        assert span.name == "test_span"
        assert span.span_id
        assert span.status == "ok"
        assert span.attributes == {}

    def test_span_duration(self):
        span = TraceSpan(name="timed")
        start = span.start_time
        span.finish()
        assert span.duration_ms >= 0
        assert span.end_time is not None
        assert span.end_time >= start

    def test_span_attributes(self):
        span = TraceSpan(name="attr_span")
        span.set_attribute("key", "value")
        span.set_attribute("count", 42)
        assert span.attributes["key"] == "value"
        assert span.attributes["count"] == 42

    def test_span_events(self):
        span = TraceSpan(name="event_span")
        span.add_event("started", {"phase": "init"})
        span.add_event("done")
        assert len(span.events) == 2
        assert span.events[0]["name"] == "started"
        assert span.events[0]["attributes"]["phase"] == "init"

    def test_span_finish_error(self):
        span = TraceSpan(name="bad")
        span.finish(status="error")
        assert span.status == "error"

    def test_span_to_dict(self):
        span = TraceSpan(name="dict_test", trace_id="abc123")
        span.set_attribute("x", 1)
        span.finish()
        d = span.to_dict()
        assert d["name"] == "dict_test"
        assert d["trace_id"] == "abc123"
        assert d["status"] == "ok"
        assert d["attributes"]["x"] == 1
        assert "duration_ms" in d


class TestEvalTracer:
    """Integration tests for EvalTracer."""

    def test_start_span(self):
        tracer = EvalTracer()
        with tracer.start_span("op") as span:
            span.set_attribute("step", 1)
        assert span.status == "ok"
        assert span.attributes["step"] == 1

    def test_nested_spans(self):
        tracer = EvalTracer()
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                pass
            assert child.parent_id == parent.span_id
            assert child.trace_id == parent.trace_id

    def test_span_error_propagation(self):
        tracer = EvalTracer()
        try:
            with tracer.start_span("failing"):
                raise ValueError("boom")
        except ValueError:
            pass
        spans = tracer._spans
        assert spans[0].status == "error"
        assert spans[0].attributes.get("error.message") == "boom"

    def test_get_trace(self):
        tracer = EvalTracer()
        tid = "trace-42"
        with tracer.start_span("s1", trace_id=tid):
            pass
        with tracer.start_span("s2", trace_id=tid):
            pass
        with tracer.start_span("s3", trace_id="other"):
            pass
        trace_spans = tracer.get_trace(tid)
        assert len(trace_spans) == 2

    def test_get_all_traces(self):
        tracer = EvalTracer()
        with tracer.start_span("a", trace_id="t1"):
            pass
        with tracer.start_span("b", trace_id="t2"):
            pass
        all_traces = tracer.get_all_traces()
        assert "t1" in all_traces
        assert "t2" in all_traces
        assert len(all_traces["t1"]) == 1

    def test_clear(self):
        tracer = EvalTracer()
        with tracer.start_span("x"):
            pass
        assert len(tracer._spans) == 1
        tracer.clear()
        assert len(tracer._spans) == 0

    def test_get_summary(self):
        tracer = EvalTracer()
        with tracer.start_span("s1"):
            time.sleep(0.001)
        with tracer.start_span("s2"):
            time.sleep(0.001)
        summary = tracer.get_summary()
        assert summary["total_spans"] == 2
        assert summary["error_spans"] == 0
        assert summary["unique_traces"] >= 1
        assert summary["avg_duration_ms"] > 0
