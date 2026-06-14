"""Trace exporters for various backends."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


class ConsoleExporter:
    """Export traces to console."""

    def export(self, traces: Dict[str, List[Dict[str, Any]]]) -> None:
        for trace_id, spans in traces.items():
            print(f"\n{'='*60}")
            print(f"Trace: {trace_id}")
            print(f"{'='*60}")
            for span in spans:
                indent = "  " * self._depth(spans, span)
                duration = span.get("duration_ms", 0)
                status = span.get("status", "ok")
                icon = "✓" if status == "ok" else "✗"
                print(f"{indent}{icon} {span['name']} ({duration:.1f}ms)")
                if span.get("attributes"):
                    for k, v in span["attributes"].items():
                        print(f"{indent}    {k}: {v}")

    def _depth(self, spans: List[Dict[str, Any]], span: Dict[str, Any]) -> int:
        """Calculate span depth in trace tree."""
        depth = 0
        parent_id = span.get("parent_id")
        while parent_id:
            parent = next((s for s in spans if s["span_id"] == parent_id), None)
            if parent:
                depth += 1
                parent_id = parent.get("parent_id")
            else:
                break
        return depth


class FileExporter:
    """Export traces to JSON file."""

    def __init__(self, output_path: str = ".areval/traces.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, traces: Dict[str, List[Dict[str, Any]]]) -> None:
        with open(self.output_path, "w") as f:
            json.dump(traces, f, indent=2, default=str)


class OTLPExporter:
    """Export spans to an OpenTelemetry Collector via OTLP/HTTP.

    Posts span data in OTLP JSON format to the configured endpoint.
    Requires ``httpx`` (already a project dependency).

    Environment variables
    ---------------------
    OTEL_EXPORTER_OTLP_ENDPOINT : str
        Base URL of the OTEL Collector (default ``http://localhost:4318``).
    OTEL_SERVICE_NAME : str
        Service name reported in resource attributes (default ``areval``).

    Usage
    -----
    >>> tracer = EvalTracer()
    >>> tracer.add_exporter(OTLPExporter())
    >>> # ... evaluation runs ...
    >>> tracer.export()  # pushes to OTEL Collector
    """

    def __init__(
        self,
        endpoint: str | None = None,
        service_name: str | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
        )
        self.service_name = service_name or os.environ.get(
            "OTEL_SERVICE_NAME", "areval"
        )

    def export(self, traces: Dict[str, List[Dict[str, Any]]]) -> None:
        """Convert internal spans to OTLP JSON and POST to collector."""
        if not traces:
            return

        resource_spans = self._build_resource_spans(traces)
        payload = {"resourceSpans": resource_spans}

        try:
            import httpx
        except ImportError:
            return  # silent no-op when httpx is unavailable

        try:
            resp = httpx.post(
                f"{self.endpoint.rstrip('/')}/v1/traces",
                json=payload,
                timeout=10.0,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        except Exception:
            pass  # best-effort — don't crash the evaluation pipeline

    def _build_resource_spans(
        self, traces: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Build OTLP-compliant resource spans from internal spans."""
        scope_spans: Dict[str, List[Dict[str, Any]]] = {}
        for trace_id, spans in traces.items():
            otlp_spans = [self._to_otlp_span(s) for s in spans]
            key = trace_id
            scope_spans[key] = otlp_spans

        return [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": self.service_name}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "areval.evaluator"},
                        "spans": otlp_spans,
                    }
                    for otlp_spans in scope_spans.values()
                ],
            }
        ]

    @staticmethod
    def _to_otlp_span(span: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an internal TraceSpan dict to OTLP span format."""
        start_ns = int(span.get("start_time", time.time()) * 1_000_000_000)
        end_ns = start_ns + int(span.get("duration_ms", 0) * 1_000_000)

        status_code = 1  # STATUS_CODE_OK
        if span.get("status") == "error":
            status_code = 2  # STATUS_CODE_ERROR

        attributes = []
        for k, v in span.get("attributes", {}).items():
            if isinstance(v, str):
                attributes.append({"key": k, "value": {"stringValue": v}})
            elif isinstance(v, (int, float)):
                attributes.append({"key": k, "value": {"doubleValue": float(v)}})
            elif isinstance(v, bool):
                attributes.append({"key": k, "value": {"boolValue": v}})

        return {
            "traceId": span.get("trace_id", ""),
            "spanId": span.get("span_id", ""),
            "parentSpanId": span.get("parent_id", ""),
            "name": span.get("name", ""),
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano": str(end_ns),
            "status": {"code": status_code},
            "attributes": attributes,
        }
