"""RAG-specific evaluation metrics.

The RAG Triad: Faithfulness, Answer Relevance, Context Precision.

Each metric supports three-tier fallback:

- **Tier 1** — heuristic mock (existing LLMJudge jaccard mock)
- **Tier 2** — local LLM (qwen3-1.7b via LM Studio)
- **Tier 3** — remote LLM (OpenAI / Anthropic / custom)

Fallback chain (``provider="auto"``):  **Tier 2 → Tier 3 → Tier 1**

Inspired by RAGAS and DeepEval's RAG metrics.
"""

from __future__ import annotations
import re
from typing import Any, Optional

from areval.metrics.base import Metric, MetricResult
from areval.routing import router
from areval.test_case import TestCase, AgentOutput

# ============================================================================
# Shared rubric templates (Tier-3 LLMJudge path — preserved unchanged)
# ============================================================================

_FAITHFULNESS_RUBRIC = """\
You are an expert factuality evaluator for Retrieval-Augmented Generation (RAG).

Judge whether the Agent's answer is **strictly grounded** in the provided
context.  Flag every claim that cannot be directly verified against the
context as potential hallucination.

Scoring guide (0.0 – 1.0):
  * 1.0 — every claim in the answer is directly supported by the context
  * 0.7 — most claims supported; a few minor, reasonable inferences
  * 0.4 — several unsupported claims or mild contradictions with the context
  * 0.1 — answer largely contradicts the context or invents facts
  * 0.0 — answer is completely fabricated, context provides zero support

Retrieved context:
{context}

Question:
{input}

Agent answer:
{actual_output}

Output format (exact):
SCORE: <0.0-1.0>
REASONING: <step-by-step analysis, mention unsupported claims if any>
"""

_ANSWER_RELEVANCE_RUBRIC = """\
You are an expert relevance evaluator.

Judge whether the Agent's answer **directly and completely** addresses the
user's question.  Penalise tangents, irrelevant details, or evasive
responses.

Scoring guide (0.0 – 1.0):
  * 1.0 — answer fully and directly addresses every part of the question
  * 0.7 — addresses the core question but misses or glosses over one aspect
  * 0.4 — partially relevant; significant sections are off-topic
  * 0.1 — mostly irrelevant or evasive
  * 0.0 — completely unrelated / does not answer the question at all

Question:
{input}

Agent answer:
{actual_output}

Output format (exact):
SCORE: <0.0-1.0>
REASONING: <analysis>
"""

_CONTEXT_PRECISION_RUBRIC = """\
You are an expert retrieval-quality evaluator.

Judge how much of the retrieved context is **genuinely useful** for
answering the question.  High precision means most of the context is
relevant; low precision means it is mostly noise.

Scoring guide (0.0 – 1.0):
  * 1.0 — virtually all of the context is directly relevant and helpful
  * 0.7 — most of the context is relevant, with some minor noise
  * 0.4 — only a small portion of the context is useful
  * 0.1 — context is almost entirely irrelevant
  * 0.0 — context has zero bearing on the question

Question:
{input}

Retrieved context:
{context}

Output format (exact):
SCORE: <0.0-1.0>
REASONING: <analysis, mention which parts of the context are useful>
"""


# ============================================================================
# Tier-2 prompt templates (for qwen3-1.7b via LocalLLMProvider)
# ============================================================================

_CONTEXT_PRECISION_T2_PROMPT = """\
Given a question and a context passage, determine if the context is relevant
to answering the question. Answer only one word: RELEVANT or NOT_RELEVANT.

Question: {question}
Context: {context}
"""

_ANSWER_RELEVANCE_T2_PROMPT = """\
Given a question and an answer, determine if the answer is relevant to the
question. Answer only one word: RELEVANT or NOT_RELEVANT.

Question: {question}
Answer: {answer}
"""

_FAITHFULNESS_T2_PROMPT = """\
Given the context and a statement, determine if the statement is supported
by the context. Answer only one word: SUPPORTED, CONTRADICTED, or NEUTRAL.

Context: {context}
Statement: {sentence}
"""

# Batch version — all sentences in one call
_FAITHFULNESS_T2_BATCH_PROMPT = """\
Given the context below, evaluate whether each numbered statement is supported
by the context. Answer for each statement with one word: SUPPORTED,
CONTRADICTED, or NEUTRAL.

Context: {context}

{statements}

Output format (one line per statement number):
1: <SUPPORTED|CONTRADICTED|NEUTRAL>
2: <SUPPORTED|CONTRADICTED|NEUTRAL>
...
"""


# ============================================================================
# Helpers
# ============================================================================

_SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？]\s*")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _normalise_label(raw: str, valid_labels: tuple[str, ...], default: str) -> str:
    """Fuzzy-match an LLM output against a set of valid labels.

    - Strips whitespace
    - Uppercases
    - Checks if the raw text *starts with* or *contains* a valid label
    """
    clean = raw.strip().upper()
    for label in valid_labels:
        if clean == label:
            return label
        if clean.startswith(label):
            return label
    # More lenient: check containment
    for label in valid_labels:
        if label in clean:
            return label
    return default


def _normalise_binary(raw: str) -> float:
    """Return 1.0 for RELEVANT, 0.0 otherwise."""
    label = _normalise_label(raw, ("RELEVANT", "NOT_RELEVANT"), "NOT_RELEVANT")
    return 1.0 if label == "RELEVANT" else 0.0


def _compute_faithfulness_score(
    llm_results: list[tuple[str, str]],  # (sentence, label)
    neutral_weight: float = 0.0,
) -> float:
    """Compute faithfulness score from per-sentence labels.

    Parameters
    ----------
    llm_results : list of (sentence, label)
        Each label is ``"SUPPORTED"``, ``"CONTRADICTED"``, or ``"NEUTRAL"``.
    neutral_weight : float
        Weight assigned to NEUTRAL labels.  Default 0.0 (conservative).
    """
    if not llm_results:
        return 1.0  # empty answer → fully faithful

    total = 0.0
    for _sentence, label in llm_results:
        if label == "SUPPORTED":
            total += 1.0
        elif label == "CONTRADICTED":
            total += 0.0
        elif label == "NEUTRAL":
            total += neutral_weight  # default 0.0 — conservative
    return total / len(llm_results)


def _parse_faithfulness_batch(raw: str, expected_count: int) -> list[str]:
    """Parse a batch faithfulness response into per-sentence labels."""
    labels: list[str] = []
    valid = {"SUPPORTED", "CONTRADICTED", "NEUTRAL"}
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Match patterns like "1: SUPPORTED", "1 SUPPORTED", "1. SUPPORTED"
        m = re.match(r"\d+[.:)\s]*\s*(.+)", line)
        if m:
            label = _normalise_label(m.group(1), tuple(valid), "NEUTRAL")
            labels.append(label)

    # Pad or truncate to expected count
    while len(labels) < expected_count:
        labels.append("NEUTRAL")
    return labels[:expected_count]


# ============================================================================
# Metrics
# ============================================================================


class FaithfulnessMetric(Metric):
    """Measures if the answer is faithful to the retrieved context.

    Three-tier evaluation:

    - **Tier 2** (default): qwen3-1.7b sentence-by-sentence judgement via
      LocalLLMProvider.
    - **Tier 3**: remote LLM via LLMJudge with rubric (complex / long-text).
    - **Tier 1**: heuristic Jaccard mock (LLMJudge fallback).

    Long-text routing: when ``len(context) + len(answer)`` exceeds
    ``complexity_threshold`` (default 2000), the metric auto-upgrades to
    Tier 3 to avoid quality degradation on 1.7B models.
    """

    name = "faithfulness"

    def __init__(
        self,
        threshold: float = 0.7,
        provider: str = "auto",
        complexity_threshold: int = 2000,
        neutral_weight: float = 0.0,
        local_llm_url: Optional[str] = None,
        local_llm_model: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, provider=provider, **kwargs)
        self.complexity_threshold = complexity_threshold
        self.neutral_weight = neutral_weight
        self._local_url = local_llm_url
        self._local_model = local_llm_model

    # -- helpers ---------------------------------------------------------------

    def _evaluate_tier2(self, context: str, answer: str) -> MetricResult:
        """Sentence-level faithfulness via qwen3-1.7b (batch mode)."""
        sentences = _split_sentences(answer)
        if not sentences:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty answer — vacuously faithful",
                threshold=self.threshold,
            )

        from areval.providers.local_llm import LocalLLMProvider

        llm = LocalLLMProvider(
            base_url=self._local_url,
            model=self._local_model,
        )

        # Build batch prompt
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        prompt = _FAITHFULNESS_T2_BATCH_PROMPT.format(
            context=context or "(none)",
            statements=numbered,
        )
        raw = llm.chat_complete(prompt)
        labels = _parse_faithfulness_batch(raw, len(sentences))
        results = list(zip(sentences, labels))
        score = _compute_faithfulness_score(results, self.neutral_weight)

        supported = sum(1 for _, label in results if label == "SUPPORTED")
        contradicted = sum(1 for _, label in results if label == "CONTRADICTED")

        return MetricResult(
            name=self.name,
            score=score,
            reasoning=(
                f"Tier-2 qwen3-1.7b: {supported}/{len(sentences)} supported, "
                f"{contradicted} contradicted"
            ),
            threshold=self.threshold,
            metadata={
                "tier": "tier2",
                "model": self._local_model or "qwen3-1.7b",
                "sentences": len(sentences),
                "supported": supported,
                "contradicted": contradicted,
            },
        )

    def _evaluate_tier3(
        self, test_case: TestCase, agent_output: AgentOutput, context: str
    ) -> MetricResult:
        """Fallback to remote LLM (Tier 3) via existing LLMJudge."""
        from areval.judges.llm_judge import LLMJudge

        judge = LLMJudge(
            rubric=_FAITHFULNESS_RUBRIC,
            criteria=["faithfulness"],
        )
        result = judge.evaluate(test_case, agent_output, context=context)
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            reasoning=result.reasoning or "Tier-3 remote LLM faithfulness evaluation",
            threshold=self.threshold,
            metadata={"tier": "tier3", "judge_provider": judge.provider},
        )

    def _evaluate_tier1(
        self, test_case: TestCase, agent_output: AgentOutput, context: str
    ) -> MetricResult:
        """Heuristic mock fallback (Tier 1)."""
        from areval.judges.llm_judge import LLMJudge

        judge = LLMJudge(
            rubric=_FAITHFULNESS_RUBRIC,
            criteria=["faithfulness"],
            provider="mock",
        )
        result = judge.evaluate(test_case, agent_output, context=context)
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            reasoning=result.reasoning or "Tier-1 heuristic faithfulness evaluation",
            threshold=self.threshold,
            metadata={"tier": "tier1"},
        )

    # -- main entry ------------------------------------------------------------

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        context = test_case.context or ""
        if not context:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No context provided — cannot assess faithfulness",
                threshold=self.threshold,
            )

        answer = agent_output.output

        # Long-text → use a different routing key (auto-upgrade to Tier 3)
        is_complex = len(context) + len(answer) > self.complexity_threshold
        task_name = "faithfulness_complex" if is_complex else "faithfulness"

        tier = router.resolve(task_name, provider=self.provider)

        if tier == "tier2":
            return self._evaluate_tier2(context, answer)
        if tier == "tier3":
            return self._evaluate_tier3(test_case, agent_output, context)
        return self._evaluate_tier1(test_case, agent_output, context)


# ============================================================================
# AnswerRelevanceMetric
# ============================================================================


class AnswerRelevanceMetric(Metric):
    """Measures if the answer is relevant to the question.

    Three-tier evaluation:

    - **Tier 2** (default): qwen3-1.7b binary classification
      (RELEVANT / NOT_RELEVANT).
    - **Tier 3**: remote LLM via LLMJudge with rubric.
    - **Tier 1**: heuristic mock (LLMJudge fallback).
    """

    name = "answer_relevance"

    def __init__(
        self,
        threshold: float = 0.7,
        provider: str = "auto",
        local_llm_url: Optional[str] = None,
        local_llm_model: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, provider=provider, **kwargs)
        self._local_url = local_llm_url
        self._local_model = local_llm_model

    # -- helpers ---------------------------------------------------------------

    def _evaluate_tier2(self, question: str, answer: str) -> MetricResult:
        from areval.providers.local_llm import LocalLLMProvider

        llm = LocalLLMProvider(base_url=self._local_url, model=self._local_model)
        prompt = _ANSWER_RELEVANCE_T2_PROMPT.format(
            question=question,
            answer=answer,
        )
        raw = llm.chat_complete(prompt)
        score = _normalise_binary(raw)

        return MetricResult(
            name=self.name,
            score=score,
            reasoning=f"Tier-2 qwen3-1.7b: {'RELEVANT' if score > 0.5 else 'NOT_RELEVANT'}",
            threshold=self.threshold,
            metadata={"tier": "tier2", "model": self._local_model or "qwen3-1.7b"},
        )

    def _evaluate_tier3(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        from areval.judges.llm_judge import LLMJudge

        judge = LLMJudge(
            rubric=_ANSWER_RELEVANCE_RUBRIC,
            criteria=["relevance"],
        )
        result = judge.evaluate(test_case, agent_output)
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            reasoning=result.reasoning or "Tier-3 remote LLM relevance evaluation",
            threshold=self.threshold,
            metadata={"tier": "tier3", "judge_provider": judge.provider},
        )

    def _evaluate_tier1(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        from areval.judges.llm_judge import LLMJudge

        judge = LLMJudge(
            rubric=_ANSWER_RELEVANCE_RUBRIC,
            criteria=["relevance"],
            provider="mock",
        )
        result = judge.evaluate(test_case, agent_output)
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            reasoning=result.reasoning or "Tier-1 heuristic relevance evaluation",
            threshold=self.threshold,
            metadata={"tier": "tier1"},
        )

    # -- main entry ------------------------------------------------------------

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        question = test_case.input.strip()
        answer = agent_output.output

        if not question:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty question — vacuously relevant",
                threshold=self.threshold,
            )

        tier = router.resolve("answer_relevance", provider=self.provider)

        if tier == "tier2":
            return self._evaluate_tier2(question, answer)
        if tier == "tier3":
            return self._evaluate_tier3(test_case, agent_output)
        return self._evaluate_tier1(test_case, agent_output)


# ============================================================================
# ContextPrecisionMetric
# ============================================================================


class ContextPrecisionMetric(Metric):
    """Measures the precision of retrieved context.

    Three-tier evaluation:

    - **Tier 2** (default): qwen3-1.7b binary classification
      (RELEVANT / NOT_RELEVANT).
    - **Tier 3**: remote LLM via LLMJudge with rubric.
    - **Tier 1**: heuristic mock (LLMJudge fallback).
    """

    name = "context_precision"

    def __init__(
        self,
        threshold: float = 0.7,
        provider: str = "auto",
        local_llm_url: Optional[str] = None,
        local_llm_model: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, provider=provider, **kwargs)
        self._local_url = local_llm_url
        self._local_model = local_llm_model

    # -- helpers ---------------------------------------------------------------

    def _evaluate_tier2(self, question: str, context: str) -> MetricResult:
        from areval.providers.local_llm import LocalLLMProvider

        llm = LocalLLMProvider(base_url=self._local_url, model=self._local_model)
        prompt = _CONTEXT_PRECISION_T2_PROMPT.format(
            question=question,
            context=context,
        )
        raw = llm.chat_complete(prompt)
        score = _normalise_binary(raw)

        return MetricResult(
            name=self.name,
            score=score,
            reasoning=(f"Tier-2 qwen3-1.7b: " f"{'RELEVANT' if score > 0.5 else 'NOT_RELEVANT'}"),
            threshold=self.threshold,
            metadata={"tier": "tier2", "model": self._local_model or "qwen3-1.7b"},
        )

    def _evaluate_tier3(
        self, test_case: TestCase, agent_output: AgentOutput, context: str
    ) -> MetricResult:
        from areval.judges.llm_judge import LLMJudge

        judge = LLMJudge(
            rubric=_CONTEXT_PRECISION_RUBRIC,
            criteria=["precision"],
        )
        result = judge.evaluate(test_case, agent_output, context=context)
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            reasoning=result.reasoning or "Tier-3 remote LLM precision evaluation",
            threshold=self.threshold,
            metadata={"tier": "tier3", "judge_provider": judge.provider},
        )

    def _evaluate_tier1(
        self, test_case: TestCase, agent_output: AgentOutput, context: str
    ) -> MetricResult:
        from areval.judges.llm_judge import LLMJudge

        judge = LLMJudge(
            rubric=_CONTEXT_PRECISION_RUBRIC,
            criteria=["precision"],
            provider="mock",
        )
        result = judge.evaluate(test_case, agent_output, context=context)
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            reasoning=result.reasoning or "Tier-1 heuristic precision evaluation",
            threshold=self.threshold,
            metadata={"tier": "tier1"},
        )

    # -- main entry ------------------------------------------------------------

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        context = test_case.context or ""
        question = test_case.input

        if not context:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No context provided",
                threshold=self.threshold,
            )

        tier = router.resolve("context_precision", provider=self.provider)

        if tier == "tier2":
            return self._evaluate_tier2(question, context)
        if tier == "tier3":
            return self._evaluate_tier3(test_case, agent_output, context)
        return self._evaluate_tier1(test_case, agent_output, context)
