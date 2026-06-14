"""RAG-specific evaluation metrics.

The RAG Triad: Faithfulness, Answer Relevance, Context Precision.
Inspired by RAGAS and DeepEval's RAG metrics.
"""

from typing import Any, List, Optional

from areval.metrics.base import Metric, MetricResult
from areval.test_case import TestCase, AgentOutput


class FaithfulnessMetric(Metric):
    """Measures if the answer is faithful to the retrieved context.

    Detects hallucinations by verifying that claims in the output
    are supported by the provided context.
    """

    name = "faithfulness"

    def __init__(self, threshold: float = 0.7, **kwargs: Any):
        super().__init__(threshold=threshold, **kwargs)

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        context = test_case.context or ""
        if not context:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No context provided for faithfulness check",
                threshold=self.threshold,
            )

        # In production: Use LLM-as-a-Judge to extract claims and verify
        # against context. For skeleton, we use a simple heuristic.
        output_lower = agent_output.output.lower()
        context_lower = context.lower()

        # Extract key phrases from output (simple word-based)
        words = set(w for w in output_lower.split() if len(w) > 4)
        context_words = set(context_lower.split())

        if not words:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty output — vacuously faithful",
                threshold=self.threshold,
            )

        supported = sum(1 for w in words if w in context_words)
        score = supported / len(words) if words else 1.0

        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=f"{supported}/{len(words)} key terms found in context",
            threshold=self.threshold,
            metadata={"supported_terms": supported, "total_terms": len(words)},
        )


class AnswerRelevanceMetric(Metric):
    """Measures if the answer is relevant to the question.

    Uses semantic similarity between the input question and the output answer.
    """

    name = "answer_relevance"

    def __init__(self, threshold: float = 0.7, **kwargs: Any):
        super().__init__(threshold=threshold, **kwargs)

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        question = test_case.input.lower()
        answer = agent_output.output.lower()

        # Simple keyword overlap as proxy for relevance
        q_words = set(w for w in question.split() if len(w) > 3)
        a_words = set(answer.split())

        if not q_words:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty question",
                threshold=self.threshold,
            )

        overlap = len(q_words & a_words)
        score = overlap / len(q_words) if q_words else 0.0

        # Boost score for non-empty answers that address the question
        if answer and len(answer) > len(question) * 0.5:
            score = min(1.0, score * 1.5)

        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=f"{overlap}/{len(q_words)} question keywords found in answer",
            threshold=self.threshold,
            metadata={"overlap": overlap, "question_keywords": len(q_words)},
        )


class ContextPrecisionMetric(Metric):
    """Measures the precision of retrieved context.

    Evaluates whether the retrieved context chunks are actually
    relevant to answering the question.
    """

    name = "context_precision"

    def __init__(self, threshold: float = 0.7, **kwargs: Any):
        super().__init__(threshold=threshold, **kwargs)

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        context = test_case.context or ""
        if not context:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No context provided",
                threshold=self.threshold,
            )

        question = test_case.input.lower()
        q_words = set(w for w in question.split() if len(w) > 3)

        # Split context into chunks (simulate retrieved chunks)
        chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
        if not chunks:
            chunks = [context]

        relevant_chunks = 0
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(q_words & chunk_words)
            if overlap > 0:
                relevant_chunks += 1

        score = relevant_chunks / len(chunks) if chunks else 0.0

        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=f"{relevant_chunks}/{len(chunks)} context chunks relevant to question",
            threshold=self.threshold,
            metadata={"chunks": len(chunks), "relevant_chunks": relevant_chunks},
        )
