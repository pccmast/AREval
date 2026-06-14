"""Trace exporters for various backends."""

import json
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

    def _depth(self, spans: List[Dict], span: Dict) -> int:
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
