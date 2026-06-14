"""Trace collection for evaluation runs.

Compatible with OpenTelemetry conventions for LLM observability.
Links with Gateway and Scheduler traces via trace IDs.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator


@dataclass
class TraceSpan:
    """A single span in an evaluation trace.

    Follows OpenTelemetry semantic conventions for LLM spans.
    """

    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    trace_id: Optional[str] = None
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "ok"  # ok, error
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def finish(self, status: str = "ok") -> None:
        self.end_time = time.time()
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class EvalTracer:
    """Tracer for evaluation runs with OpenTelemetry compatibility.

    Key features:
    - Correlates with Gateway traces via trace_id
    - Correlates with Scheduler traces via span context
    - Captures metric execution traces
    - Exports to multiple backends
    """

    def __init__(self, service_name: str = "areval"):
        self.service_name = service_name
        self._spans: List[TraceSpan] = []
        self._span_stack: List[TraceSpan] = []
        self._exporters: List[Any] = []

    def add_exporter(self, exporter: Any) -> None:
        """Add an exporter for trace data."""
        self._exporters.append(exporter)

    @contextmanager
    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[TraceSpan, None, None]:
        """Start a new trace span.

        Usage:
            with tracer.start_span("metric_execution", attributes={"metric": "exact_match"}) as span:
                result = metric.measure(test_case, output)
                span.set_attribute("score", result.score)
        """
        parent = self._span_stack[-1] if self._span_stack else None
        span = TraceSpan(
            name=name,
            trace_id=trace_id or (parent.trace_id if parent else str(uuid.uuid4())),
            parent_id=parent.span_id if parent else None,
            attributes=attributes or {},
        )

        self._spans.append(span)
        self._span_stack.append(span)

        try:
            yield span
            span.finish(status="ok")
        except Exception as e:
            span.finish(status="error")
            span.set_attribute("error.message", str(e))
            raise
        finally:
            self._span_stack.pop()

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        return [s for s in self._spans if s.trace_id == trace_id]

    def export(self) -> None:
        """Export all traces to configured exporters."""
        traces: Dict[str, List[Dict[str, Any]]] = {}
        for span in self._spans:
            tid = span.trace_id or "unknown"
            if tid not in traces:
                traces[tid] = []
            traces[tid].append(span.to_dict())

        for exporter in self._exporters:
            try:
                exporter.export(traces)
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all traces."""
        self._spans.clear()
        self._span_stack.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get trace summary statistics."""
        total_spans = len(self._spans)
        error_spans = sum(1 for s in self._spans if s.status == "error")
        traces = len(set(s.trace_id for s in self._spans if s.trace_id))

        return {
            "total_spans": total_spans,
            "error_spans": error_spans,
            "unique_traces": traces,
            "avg_duration_ms": sum(s.duration_ms for s in self._spans) / total_spans if total_spans else 0,
        }
