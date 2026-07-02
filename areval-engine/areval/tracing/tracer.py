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
    conversation_id: Optional[str] = None   # multi-turn conversation grouping
    turn_index: int = 0                      # turn number within conversation
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
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
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
        # Multi-turn conversation tracking
        self._active_conversation: Optional[str] = None
        self._conversation_turn: int = 0
        self._conversations: Dict[str, List[str]] = {}  # conv_id → [trace_id, ...]

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
            conversation_id=self._active_conversation,
            turn_index=self._conversation_turn if self._active_conversation else 0,
            attributes=attributes or {},
        )

        self._spans.append(span)

        # Track conversation membership
        if self._active_conversation:
            self._conversations.setdefault(self._active_conversation, []).append(span.span_id)

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

    def get_all_traces(self) -> Dict[str, List[TraceSpan]]:
        """Return all collected spans grouped by trace_id.

        Used by :class:`~areval.datasets.curator.TraceCurator` to
        analyse trace data and auto-curate test sets.
        """
        traces: Dict[str, List[TraceSpan]] = {}
        for span in self._spans:
            tid = span.trace_id or "unknown"
            traces.setdefault(tid, []).append(span)
        return traces

    # ------------------------------------------------------------------
    # Multi-turn conversation support
    # ------------------------------------------------------------------

    def start_conversation(self, conv_id: Optional[str] = None) -> str:
        """Begin (or switch to) a multi-turn conversation.

        Subsequent :meth:`start_span` calls will automatically inherit the
        *conversation_id* and increment the *turn_index*.

        Idempotent when called with the same *conv_id* repeatedly — turn
        counter is preserved so long-running conversations don't reset.

        Returns the *conv_id* so it can be stored on the caller side.
        """
        cid = conv_id or f"conv-{uuid.uuid4().hex[:8]}"
        if self._active_conversation == cid:
            return cid  # already active — no-op, preserve turn counter
        if self._active_conversation:
            self.end_conversation()
        self._active_conversation = cid
        self._conversation_turn = 0
        self._conversations.setdefault(cid, [])
        return cid

    def next_turn(self) -> int:
        """Advance to the next turn in the active conversation.

        Returns the new *turn_index* (0-based).  Call this between
        Agent invocations when the user provides a new utterance.
        """
        self._conversation_turn += 1
        return self._conversation_turn

    def end_conversation(self) -> None:
        """End the active conversation."""
        self._active_conversation = None
        self._conversation_turn = 0

    def get_conversation_spans(self, conv_id: str) -> List[TraceSpan]:
        """Return all spans belonging to a conversation, ordered by turn."""
        span_ids = self._conversations.get(conv_id, [])
        id_set = set(span_ids)
        return sorted(
            [s for s in self._spans if s.span_id in id_set],
            key=lambda s: s.start_time,
        )

    def get_all_conversations(self) -> Dict[str, List[TraceSpan]]:
        """Return all multi-turn conversations grouped by conv_id."""
        result: Dict[str, List[TraceSpan]] = {}
        for cid in self._conversations:
            result[cid] = self.get_conversation_spans(cid)
        return result
