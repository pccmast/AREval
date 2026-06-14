"""Semantic similarity metrics using embeddings."""

from typing import Any, Optional

import numpy as np

from areval.metrics.base import Metric, MetricResult
from areval.test_case import TestCase, AgentOutput


class SemanticSimilarityMetric(Metric):
    """Semantic similarity using vector embeddings.

    Measures meaning similarity regardless of exact wording.
    Requires an embedding model — defaults to cosine similarity
    on pre-computed or API-fetched embeddings.

    Inspired by DeepEval's semantic similarity and RAGAS approaches.
    """

    name = "semantic_similarity"

    def __init__(
        self,
        threshold: float = 0.7,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self._embedding_cache: dict = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Placeholder: in production, call OpenAI/Anthropic embedding API
        # For the project skeleton, we simulate with a hash-based vector
        # In real implementation:
        #   from openai import OpenAI
        #   client = OpenAI()
        #   response = client.embeddings.create(input=text, model=self.embedding_model)
        #   vector = np.array(response.data[0].embedding)
        np.random.seed(hash(text) % (2**32))
        vector = np.random.randn(1536)
        vector = vector / np.linalg.norm(vector)
        self._embedding_cache[text] = vector
        return vector

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        if test_case.expected_output is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="No expected_output for semantic comparison",
                threshold=self.threshold,
            )

        expected_embedding = self._get_embedding(test_case.expected_output)
        actual_embedding = self._get_embedding(agent_output.output)

        similarity = self._cosine_similarity(expected_embedding, actual_embedding)
        # Normalize to [0, 1] (cosine similarity is [-1, 1])
        score = (similarity + 1) / 2

        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=f"Cosine similarity: {similarity:.4f} (normalized score: {score:.4f})",
            threshold=self.threshold,
            metadata={
                "cosine_similarity": similarity,
                "embedding_model": self.embedding_model,
            },
        )
