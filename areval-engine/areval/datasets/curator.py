"""Trace curation engine — automatically discover test cases from production data.

Analyses collected trace data to identify valuable scenarios and convert
them into :class:`~areval.test_case.TestCase` instances for regression testing.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from areval.datasets.manager import Dataset
from areval.test_case import TestCase
from areval.tracing.analyzers import TraceAnalyzer, TraceAnalysis

if TYPE_CHECKING:
    from areval.tracing.tracer import EvalTracer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CurationConfig:
    """Configuration for :class:`TraceCurator`.

    Parameters
    ----------
    min_value_score : float
        Minimum value score for a trace to be considered for curation.
    max_cases : int
        Maximum number of test cases to curate.
    dedup_similarity : float
        Jaccard similarity threshold above which two inputs are considered
        duplicates and one is discarded.
    strip_pii : bool
        **Always True** — PII stripping is mandatory and cannot be disabled
        for production data.
    include_categories : List[str]
        Only traces in these categories are curated.
    """

    min_value_score: float = 0.3
    max_cases: int = 100
    dedup_similarity: float = 0.8
    strip_pii: bool = True
    include_categories: List[str] = field(
        default_factory=lambda: ["low_score", "error", "edge_case"]
    )

    def __post_init__(self) -> None:
        if not self.strip_pii:
            raise ValueError("PII stripping must remain enabled for trace curation.")


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------


class TraceCurator:
    """Curate test cases from production trace data.

    Parameters
    ----------
    analyzer : TraceAnalyzer, optional
    config : CurationConfig, optional
    """

    def __init__(
        self,
        analyzer: Optional[TraceAnalyzer] = None,
        config: Optional[CurationConfig] = None,
    ) -> None:
        self.analyzer = analyzer or TraceAnalyzer()
        self.config = config or CurationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def curate_from_traces(
        self,
        traces: Dict[str, List[Any]],  # trace_id -> List[TraceSpan]
        existing_dataset: Optional[Dataset] = None,
    ) -> Dataset:
        """Curate a Dataset from raw trace data.

        Steps
        -----
        1. Analyse all traces with :class:`TraceAnalyzer`
        2. Filter by category and minimum value score
        3. Sort by value score (descending)
        4. De-duplicate by input-text similarity
        5. Strip PII
        6. Convert to :class:`TestCase` instances
        7. Build and return a :class:`Dataset`
        """
        # 1. Analyse
        all_analyses = self.analyzer.analyze_all(traces)

        # 2. Filter
        candidates = [
            a for a in all_analyses
            if a.category in self.config.include_categories
            and a.value_score >= self.config.min_value_score
            and a.input_text
        ]

        # 3. Sort
        candidates.sort(key=lambda a: a.value_score, reverse=True)

        # 4. Dedup
        candidates = self._dedup(candidates, threshold=self.config.dedup_similarity)

        # 5. PII strip (always enabled)
        for a in candidates:
            a.input_text = self._strip_pii(a.input_text or "")
            if a.output_text:
                a.output_text = self._strip_pii(a.output_text)

        # 6. Convert
        test_cases = [self._analysis_to_test_case(a) for a in candidates[: self.config.max_cases]]

        # 7. Build Dataset
        return Dataset(
            name=f"curated-{uuid.uuid4().hex[:8]}",
            description=f"Auto-curated from {len(all_analyses)} traces",
            test_cases=test_cases,
            tags=["auto-curated"],
        )

    def curate_from_eval_tracer(
        self,
        tracer: "EvalTracer",
        existing_dataset: Optional[Dataset] = None,
    ) -> Dataset:
        """Shortcut: curate directly from an :class:`EvalTracer` instance."""
        traces = tracer.get_all_traces()
        return self.curate_from_traces(traces, existing_dataset)

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def _dedup(
        self,
        analyses: List[TraceAnalysis],
        threshold: float = 0.8,
    ) -> List[TraceAnalysis]:
        """Greedy dedup by Jaccard similarity of tokenised input text.

        Complexity: O(n²), acceptable for n < 1000.
        """
        kept: List[TraceAnalysis] = []
        for a in analyses:
            text_a = self._tokenize(a.input_text or "")
            if not text_a:
                kept.append(a)
                continue
            if any(
                self._jaccard(text_a, self._tokenize(b.input_text or "")) >= threshold
                for b in kept
            ):
                continue
            kept.append(a)
        return kept

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # PII  (privacy is non-negotiable)
    # ------------------------------------------------------------------

    def _strip_pii(self, text: str) -> str:
        """Remove common PII patterns from *text*."""
        # Email
        text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
        # Phone (Chinese / international)
        text = re.sub(r"1[3-9]\d{9}", "[PHONE]", text)
        text = re.sub(r"\+\d{1,3}[\s-]?\d{4,14}", "[PHONE]", text)
        # ID card  (18-digit Chinese)
        text = re.sub(r"\b\d{17}[\dXx]\b", "[ID_CARD]", text)
        return text

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _analysis_to_test_case(self, analysis: TraceAnalysis) -> TestCase:
        """Convert a TraceAnalysis into a TestCase.

        .. note::
           ``expected_output`` is set to ``None`` because automatic
           curation cannot generate a ground-truth answer.  Use
           LLM-as-a-Judge for evaluating these cases.
        """
        return TestCase(
            name=f"curated-{analysis.trace_id[:8]}",
            input=analysis.input_text or "",
            expected_output=analysis.output_text,  # kept for reference, not as ground truth
            tags=["auto-curated", analysis.category],
            metadata={
                "source_trace_id": analysis.trace_id,
                "value_score": analysis.value_score,
                "curation_category": analysis.category,
            },
        )
