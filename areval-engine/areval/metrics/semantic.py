"""Semantic similarity metrics using embeddings."""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from areval.metrics.base import Metric, MetricResult
from areval.test_case import AgentOutput, TestCase


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a normalized embedding vector for the given text."""
        ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by OpenAI's embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install with: pip install openai"
            ) from e

        client = OpenAI(api_key=self.api_key, timeout=60.0, max_retries=3)
        response = client.embeddings.create(input=text, model=self.model)
        vector = np.array(response.data[0].embedding, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        self._cache[text] = vector
        return vector


class OfflineEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline embedding provider for CI and demo use.

    Generates a pseudo-random unit vector from the hash of the input text.
    This is NOT semantically meaningful and is only intended as a fallback
    when no embedding API key is available.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        rng = np.random.RandomState(hash(text) % (2**32))
        vector = rng.randn(self.dimension)
        vector = vector / np.linalg.norm(vector)

        self._cache[text] = vector
        return vector


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
        self.embedding_provider_name = embedding_provider
        self.embedding_model = embedding_model
        self._embedding_provider = self._create_provider(
            embedding_provider, embedding_model, kwargs
        )

    def _create_provider(
        self,
        provider_name: str,
        model: str,
        kwargs: dict[str, Any],
    ) -> EmbeddingProvider:
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")

        if provider_name == "openai" and api_key:
            return OpenAIEmbeddingProvider(model=model, api_key=api_key)

        if provider_name == "openai" and not api_key:
            print(
                "[SemanticSimilarityMetric] Warning: OPENAI_API_KEY not found. "
                "Falling back to offline (deterministic) embeddings."
            )

        if provider_name not in ("openai", "offline"):
            print(
                f"[SemanticSimilarityMetric] Warning: unknown provider {provider_name!r}. "
                "Falling back to offline embeddings."
            )

        return OfflineEmbeddingProvider()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        return self._embedding_provider.embed(text)

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
                "embedding_provider": self.embedding_provider_name,
            },
        )
