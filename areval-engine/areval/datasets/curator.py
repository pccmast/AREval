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
        Maximum number of test cases to curate (fixed ceiling; kept for
        backward compatibility).
    max_ratio : float
        Fraction of candidates to retain (0 < ratio <= 1).  Used together
        with *min_keep* and *absolute_max* to compute the effective limit.
    min_keep : int
        Floor on curated count — useful when total traces are few but
        high-quality.
    absolute_max : int
        Hard upper bound (prevents memory blow-up on huge trace pools).
    dedup_similarity : float
        Jaccard similarity threshold above which two inputs are considered
        duplicates and one is discarded.
    strip_pii : bool
        **Always True** — PII stripping is mandatory and cannot be disabled
        for production data.
    include_categories : List[str]
        Only traces in these categories are curated.
    require_review : bool
        When True (default), curated cases are tagged ``"pending_review"``
        and excluded from evaluation until a human approves them.
    """

    min_value_score: float = 0.3
    max_cases: int = 100
    max_ratio: float = 0.2
    min_keep: int = 20
    absolute_max: int = 500
    dedup_similarity: float = 0.8
    strip_pii: bool = True
    include_categories: List[str] = field(
        default_factory=lambda: ["low_score", "error", "edge_case"]
    )
    require_review: bool = True

    def __post_init__(self) -> None:
        if not self.strip_pii:
            raise ValueError("PII stripping must remain enabled for trace curation.")
        if not (0 < self.max_ratio <= 1.0):
            raise ValueError("max_ratio must be in (0, 1]")
        if self.min_keep < 1:
            raise ValueError("min_keep must be >= 1")
        if self.absolute_max < self.min_keep:
            raise ValueError("absolute_max must be >= min_keep")


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

    def _effective_max(self, candidate_count: int) -> int:
        """Compute the dynamic curation limit.

        Formula: ``max(min_keep, min(absolute_max, candidate_count * max_ratio))``
        This replaces the old fixed *max_cases* ceiling when *candidate_count*
        is substantially larger than *min_keep*.
        """
        return max(
            self.config.min_keep,
            min(self.config.absolute_max, int(candidate_count * self.config.max_ratio)),
        )

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

        # 4. Dedup — use the dynamic limit to bound comparisons
        effective_max = self._effective_max(len(candidates))
        candidates = self._dedup(
            candidates,
            threshold=self.config.dedup_similarity,
            max_keep=effective_max,
        )

        # 5. PII strip (always enabled)
        for a in candidates:
            a.input_text = self._strip_pii(a.input_text or "")
            if a.output_text:
                a.output_text = self._strip_pii(a.output_text)

        # 6. Convert (respect dynamic limit)
        effective_max = self._effective_max(len(candidates))
        test_cases = [self._analysis_to_test_case(a) for a in candidates[: effective_max]]

        # 7. Build Dataset
        tags = ["auto-curated"]
        if self.config.require_review:
            tags.append("pending_review")
        return Dataset(
            name=f"curated-{uuid.uuid4().hex[:8]}",
            description=f"Auto-curated from {len(all_analyses)} traces",
            test_cases=test_cases,
            tags=tags,
        )

    def curate_from_eval_tracer(
        self,
        tracer: "EvalTracer",
        existing_dataset: Optional[Dataset] = None,
    ) -> Dataset:
        """Shortcut: curate directly from an :class:`EvalTracer` instance."""
        # Multi-turn conversations take priority if any exist
        conversations = tracer.get_all_conversations()
        if conversations:
            return self.curate_conversations(conversations)
        traces = tracer.get_all_traces()
        return self.curate_from_traces(traces, existing_dataset)

    def curate_conversations(
        self,
        conversations: Dict[str, List[Any]],
    ) -> Dataset:
        """Curate a Dataset from multi-turn conversation traces.

        Each conversation (grouped by *conversation_id*) becomes one
        ``TestCase`` whose *input* is the concatenated turns and
        *context* is the full dialogue history.
        """
        all_analyses: list = []
        for conv_id, spans in conversations.items():
            if not spans:
                continue
            analysis = self.analyzer.analyze_conversation(conv_id, spans)
            all_analyses.append(analysis)

        candidates = [
            a for a in all_analyses
            if a.category in self.config.include_categories
            and a.value_score >= self.config.min_value_score
            and a.input_text
        ]
        candidates.sort(key=lambda a: a.value_score, reverse=True)

        effective_max = self._effective_max(len(candidates))
        candidates = self._dedup(candidates, threshold=self.config.dedup_similarity, max_keep=effective_max)

        for a in candidates:
            a.input_text = self._strip_pii(a.input_text or "")
            if a.output_text:
                a.output_text = self._strip_pii(a.output_text)

        effective_max = self._effective_max(len(candidates))
        test_cases = [self._analysis_to_test_case(a) for a in candidates[:effective_max]]

        return Dataset(
            name=f"curated-conv-{uuid.uuid4().hex[:8]}",
            description=f"Auto-curated from {len(all_analyses)} conversations",
            test_cases=test_cases,
            tags=["auto-curated", "multi-turn"],
        )

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def _dedup(
        self,
        analyses: List[TraceAnalysis],
        threshold: float = 0.8,
        max_keep: int = 100,
    ) -> List[TraceAnalysis]:
        """Greedy dedup by Jaccard similarity with early termination.

        Only compares each candidate against already-kept items.  Stops
        as soon as ``max_keep`` items are selected, bounding complexity
        to O(n · max_keep) instead of O(n²).
        """
        kept: List[TraceAnalysis] = []
        for a in analyses:
            text_a = self._tokenize(a.input_text or "")
            if not text_a:
                kept.append(a)
                if len(kept) >= max_keep:
                    break
                continue
            if any(
                self._jaccard(text_a, self._tokenize(b.input_text or "")) >= threshold
                for b in kept
            ):
                continue
            kept.append(a)
            if len(kept) >= max_keep:
                break
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
        """Remove common PII patterns from *text*.

        Covers both English and Chinese PII so that automatically curated
        test sets do not leak personal data into regression baselines.
        """
        # ── English / international ──────────────────────────────
        # Email
        text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
        # International phone
        text = re.sub(r"\+\d{1,3}[\s-]?\d{4,14}", "[PHONE]", text)

        # ── Chinese ──────────────────────────────────────────────
        # Mobile phone (11 digits starting with 1)
        text = re.sub(r"1[3-9]\d{9}", "[PHONE]", text)
        # Landline with area code  (0XXX-XXXXXXXX or 0XXX XXXXXXXX)
        text = re.sub(r"0\d{2,3}[- ]?\d{7,8}", "[PHONE]", text)
        # ID card — 18-digit (last digit may be X)
        text = re.sub(r"\b\d{17}[\dXx]\b", "[ID_CARD]", text)
        # ID card — legacy 15-digit
        text = re.sub(r"\b\d{15}\b", "[ID_CARD]", text)
        # Chinese name — surname (1-2 chars) + given name (1-2 chars)
        # Common surnames (百家姓 top ~120) grouped for efficiency
        _CN_SURNAME = (
            r"[王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林郑梁谢唐许"
            r"冯宋韩邓彭曹曾田萧潘袁蔡蒋余于杜叶程苏魏吕丁任卢姚"
            r"沈钟姜崔谭陆范汪廖石金韦贾夏付方白邹孟熊秦邱江尹薛"
            r"闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
            r"温康施文牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹"
            r"申欧耿关兰芦俞]"
        )
        # Name pattern: surname + 1-2 chars (Chinese character range)
        _cn_name_re = (
            _CN_SURNAME
            + r"[\u4e00-\u9fff]{1,2}(?=[\s，。！？、；：""''）】》\\]])"
        )
        text = re.sub(_cn_name_re, "[NAME]", text)
        # Address — province/city prefix + road/street/lane + number
        text = re.sub(
            r"(?:[\u4e00-\u9fff]{2,}(?:省|市|区|县|镇|乡|村|街道|路|巷|弄|号|楼|室|单元|栋)\s*)+\d*",
            "[ADDRESS]",
            text,
        )
        # QQ number (5-11 digits)
        text = re.sub(r"\b[1-9]\d{4,10}\b", "[QQ]", text)
        # WeChat ID (字母数字下划线减号组合)
        text = re.sub(r"(?i)(?:weixin|wechat|wx|微信)[:：]?\s*[a-zA-Z][a-zA-Z0-9_-]{5,19}", "[WECHAT]", text)
        # Bank card number (16 or 19 digits)
        text = re.sub(r"\b\d{16}\b", "[BANK_CARD]", text)
        text = re.sub(r"\b\d{19}\b", "[BANK_CARD]", text)
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
        tags = ["auto-curated", analysis.category]
        if self.config.require_review:
            tags.append("pending_review")
        return TestCase(
            name=f"curated-{analysis.trace_id[:8]}",
            input=analysis.input_text or "",
            expected_output=analysis.output_text,  # kept for reference, not as ground truth
            tags=tags,
            metadata={
                "source_trace_id": analysis.trace_id,
                "value_score": analysis.value_score,
                "curation_category": analysis.category,
            },
        )
