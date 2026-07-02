"""Trace analysis for automatic test-set curation.

Analyses collected :class:`TraceSpan` data to assign a *value score*
to each trace, indicating how worthwhile it is to convert into a
:class:`~areval.test_case.TestCase`.

Scoring factors
---------------
- **Low-score factor** (weight 0.4): If the trace contains an evaluation
  score below *score_threshold*, it is more valuable for regression testing.
- **Error factor** (weight 0.3): Traces with error spans are valuable
  for catching failure modes.
- **Edge-case factor** (weight 0.2): Traces with unusually high latency
  or atypical tool-call counts.
- **Diversity factor** (weight 0.1): Traces with input text that differs
  substantially from existing test cases (reserved for future use; always
  returns 0.5, i.e. neutral, when no existing dataset is provided).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from areval.tracing.tracer import TraceSpan


@dataclass
class TraceAnalysis:
    """Analysis result for a single trace."""

    trace_id: str
    spans: List[TraceSpan] = field(default_factory=list)
    value_score: float = 0.0
    category: str = "normal"  # low_score | error | edge_case | normal
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraceAnalyzer:
    """Analyse trace data and compute value scores.

    Parameters
    ----------
    low_score_weight : float
    error_weight : float
    edge_case_weight : float
    diversity_weight : float
    score_threshold : float
    latency_percentile : float
    """

    def __init__(
        self,
        low_score_weight: float = 0.4,
        error_weight: float = 0.3,
        edge_case_weight: float = 0.2,
        diversity_weight: float = 0.1,
        score_threshold: float = 0.7,
        latency_percentile: float = 95.0,
    ) -> None:
        self.low_score_weight = low_score_weight
        self.error_weight = error_weight
        self.edge_case_weight = edge_case_weight
        self.diversity_weight = diversity_weight
        self.score_threshold = score_threshold
        self.latency_percentile = latency_percentile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_trace(self, trace_id: str, spans: List[TraceSpan]) -> TraceAnalysis:
        """Analyse a single trace."""
        analysis = TraceAnalysis(trace_id=trace_id, spans=spans)

        # Extract metadata
        analysis.input_text, analysis.output_text = self._extract_input_output(spans)
        analysis.tool_calls = self._extract_tool_calls(spans)
        analysis.error_count = sum(1 for s in spans if s.status == "error")
        analysis.latency_ms = sum(s.duration_ms for s in spans)

        # Compute value score
        analysis.value_score = self._compute_value_score(analysis)

        # Classify
        analysis.category = self._classify(analysis)

        return analysis

    def analyze_all(
        self, traces: Dict[str, List[TraceSpan]]
    ) -> List[TraceAnalysis]:
        """Analyse all traces, returning results sorted by *value_score* descending."""
        results = [self.analyze_trace(tid, spans) for tid, spans in traces.items()]
        results.sort(key=lambda a: a.value_score, reverse=True)
        return results

    def analyze_conversation(
        self, conversation_id: str, spans: List[TraceSpan]
    ) -> TraceAnalysis:
        """Analyse a multi-turn conversation as a single evaluation unit.

        Merges all turns into a single *input_text* and evaluates the
        conversation holistically (e.g. task completion across turns).
        """
        analysis = TraceAnalysis(trace_id=conversation_id, spans=spans)

        # Merge inputs from all turns
        sorted_spans = sorted(spans, key=lambda s: s.turn_index)
        analysis.input_text = "\n".join(
            f"Turn {s.turn_index}: {s.attributes.get('input.args', '')}"
            for s in sorted_spans
        )[:2000]
        # Use the last turn's output as reference
        analysis.output_text = (
            sorted_spans[-1].attributes.get("output", "") if sorted_spans else ""
        )[:1000]
        analysis.tool_calls = self._extract_tool_calls(spans)
        analysis.error_count = sum(1 for s in spans if s.status == "error")
        analysis.latency_ms = sum(s.duration_ms for s in spans)

        analysis.value_score = self._compute_value_score(analysis)
        analysis.category = self._classify(analysis)

        analysis.metadata["turn_count"] = len(spans)
        analysis.metadata["conversation_id"] = conversation_id

        return analysis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_value_score(self, analysis: TraceAnalysis) -> float:
        """Weighted sum of scoring factors."""
        factors: List[float] = []

        # 1. Low-score factor
        score = self._extract_eval_score(analysis.spans)
        if score is not None:
            factors.append(self.low_score_weight * max(0.0, 1.0 - score / self.score_threshold))

        # 2. Error factor
        factors.append(self.error_weight * min(1.0, analysis.error_count / 3.0))

        # 3. Edge-case factor — based on latency (simplified: linear ramp)
        latency_factor = min(1.0, analysis.latency_ms / 2000.0)  # 2s = max edge
        factors.append(self.edge_case_weight * latency_factor)

        # 4. Diversity factor — neutral when no existing dataset
        factors.append(self.diversity_weight * 0.5)

        return min(1.0, sum(factors))

    def _extract_eval_score(self, spans: List[TraceSpan]) -> Optional[float]:
        """Try to find an evaluation score in span attributes."""
        for span in spans:
            for key in ("score", "eval.score", "overall_score"):
                val = span.attributes.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
        return None

    def _extract_input_output(
        self, spans: List[TraceSpan],
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract input/output text from the root span."""
        root = spans[0] if spans else None
        if root is None:
            return None, None

        inp = None
        for key in ("input.args", "input", "input.text"):
            val = root.attributes.get(key)
            if isinstance(val, str) and val:
                inp = val[:500]
                break

        out = None
        for key in ("output", "output.text", "response"):
            val = root.attributes.get(key)
            if isinstance(val, str) and val:
                out = val[:500]
                break

        # Fallback: last span's output
        if out is None and len(spans) > 1:
            last = spans[-1]
            out_val = last.attributes.get("output")
            if isinstance(out_val, str):
                out = out_val[:500]

        return inp, out

    def _extract_tool_calls(self, spans: List[TraceSpan]) -> List[Dict[str, Any]]:
        """Identify tool-invocation spans."""
        tools: List[Dict[str, Any]] = []
        for span in spans:
            if "tool" in span.name.lower() or "tool.name" in span.attributes:
                tools.append({
                    "span_name": span.name,
                    "span_id": span.span_id,
                    "status": span.status,
                    "duration_ms": span.duration_ms,
                    "attributes": {k: v for k, v in span.attributes.items()
                                   if k not in ("input.args", "input.kwargs", "output")},
                })
        return tools

    @staticmethod
    def _classify(analysis: TraceAnalysis) -> str:
        if analysis.value_score >= 0.7 and analysis.error_count > 0:
            return "error"
        if analysis.value_score >= 0.5:
            return "low_score"
        if analysis.latency_ms > 1500:
            return "edge_case"
        return "normal"
