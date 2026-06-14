"""OpenTelemetry-compatible tracing for agent evaluation.

Provides trace correlation between Agent Gateway, LLM Scheduling,
and AREval for end-to-end observability.
"""

from areval.tracing.tracer import EvalTracer, TraceSpan
from areval.tracing.exporters import ConsoleExporter, FileExporter

__all__ = ["EvalTracer", "TraceSpan", "ConsoleExporter", "FileExporter"]
