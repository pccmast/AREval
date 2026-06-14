"""RAG-specific evaluation metrics.

The RAG Triad: Faithfulness, Answer Relevance, Context Precision.

Each metric delegates to an :class:`~areval.judges.llm_judge.LLMJudge`
with a specialised rubric.  When no LLM API key is available the judge
falls back to a heuristic mock, giving *strictly better* results than the
previous keyword-overlap skeleton — and true semantic evaluation when an
API key is configured.

Inspired by RAGAS and DeepEval's RAG metrics.
"""

from __future__ import annotations

from typing import Any

from areval.metrics.base import Metric, MetricResult
from areval.test_case import TestCase, AgentOutput


# ---------------------------------------------------------------------------
# Shared rubric templates
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class FaithfulnessMetric(Metric):
    """Measures if the answer is faithful to the retrieved context.

    Uses an :class:`~areval.judges.llm_judge.LLMJudge` to detect
    hallucinations by verifying that every claim in the output is supported
    by the provided context.

    Falls back to a heuristic (Jaccard-based) mock when no LLM API key is
    configured.
    """

    name = "faithfulness"

    def __init__(self, threshold: float = 0.7, **kwargs: Any):
        super().__init__(threshold=threshold, **kwargs)
        # Lazy import so that scipy / openai are not required at import time
        from areval.judges.llm_judge import LLMJudge

        self._judge = LLMJudge(
            rubric=_FAITHFULNESS_RUBRIC,
            criteria=["faithfulness"],
            **kwargs,
        )

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        context = test_case.context or ""
        if not context:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No context provided — cannot assess faithfulness",
                threshold=self.threshold,
            )

        result = self._judge.evaluate(test_case, agent_output, context=context)

        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            passed=result.score >= self.threshold,
            reasoning=result.reasoning or "LLM-based faithfulness evaluation",
            threshold=self.threshold,
            metadata={
                "judge_provider": self._judge.provider,
                "judge_model": self._judge.model,
            },
        )


class AnswerRelevanceMetric(Metric):
    """Measures if the answer is relevant to the question.

    Uses an :class:`~areval.judges.llm_judge.LLMJudge` to assess semantic
    relevance rather than simple keyword overlap.
    """

    name = "answer_relevance"

    def __init__(self, threshold: float = 0.7, **kwargs: Any):
        super().__init__(threshold=threshold, **kwargs)
        from areval.judges.llm_judge import LLMJudge

        self._judge = LLMJudge(
            rubric=_ANSWER_RELEVANCE_RUBRIC,
            criteria=["relevance"],
            **kwargs,
        )

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        if not test_case.input.strip():
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty question — vacuously relevant",
                threshold=self.threshold,
            )

        result = self._judge.evaluate(test_case, agent_output)

        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            passed=result.score >= self.threshold,
            reasoning=result.reasoning or "LLM-based relevance evaluation",
            threshold=self.threshold,
            metadata={
                "judge_provider": self._judge.provider,
                "judge_model": self._judge.model,
            },
        )


class ContextPrecisionMetric(Metric):
    """Measures the precision of retrieved context.

    Uses an :class:`~areval.judges.llm_judge.LLMJudge` to evaluate whether
    the retrieved context chunks are actually relevant to the question.
    """

    name = "context_precision"

    def __init__(self, threshold: float = 0.7, **kwargs: Any):
        super().__init__(threshold=threshold, **kwargs)
        from areval.judges.llm_judge import LLMJudge

        self._judge = LLMJudge(
            rubric=_CONTEXT_PRECISION_RUBRIC,
            criteria=["precision"],
            **kwargs,
        )

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        context = test_case.context or ""
        if not context:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No context provided",
                threshold=self.threshold,
            )

        # Context-precision does not need the agent output;
        # we still pass the agent_output (required by Judge.evaluate).
        result = self._judge.evaluate(test_case, agent_output, context=context)

        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            passed=result.score >= self.threshold,
            reasoning=result.reasoning or "LLM-based context precision evaluation",
            threshold=self.threshold,
            metadata={
                "judge_provider": self._judge.provider,
                "judge_model": self._judge.model,
            },
        )
