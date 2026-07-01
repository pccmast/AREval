"""Per-task tier routing table (ADR-6).

Determines which Tier (local model / remote LLM / pure code) each
evaluation task uses by default.  Metrics delegate tier resolution to
:class:`TaskRouter` instead of hard-coding if-else fallback chains.

Usage::

    from areval.routing import router

    tier = router.resolve("faithfulness", provider="auto")
    # → "tier2" if local LLM is available, else "tier3", else "tier1"
"""

from __future__ import annotations

import os
from typing import Optional

from areval.config import log_degradation

# ---------------------------------------------------------------------------
# Default routing table
# ---------------------------------------------------------------------------
DEFAULT_TASK_ROUTING: dict[str, str] = {
    # === RAG ===
    "context_precision":     "tier2",   # qwen3-1.7b binary classification
    "answer_relevance":      "tier2",   # qwen3-1.7b binary classification
    "faithfulness":          "tier2",   # qwen3-1.7b sentence-by-sentence
    "faithfulness_complex":  "tier3",   # remote LLM (long-text > 2000 chars)

    # === Safety ===
    "injection_resistance":  "tier2",   # qwen3-1.7b refusal detection
    "harmful_content":       "tier2",   # qwen3-1.7b multi-label classification
    "jailbreak_resistance":  "tier3",   # remote LLM (complex reasoning chains)

    # === Agent ===
    "tool_call_semantic":    "tier2",   # qwen3-1.7b semantic matching
    "task_completion_det":   "tier1",   # pure code (output-length heuristic)
    "task_completion_open":  "tier3",   # remote LLM (open-ended judgement)

    # === Other ===
    "trajectory_evaluation": "tier3",   # remote LLM (complex trajectory analysis)
}


# ---------------------------------------------------------------------------
# Tier → availability check mapping
# (lazy — first call imports and caches)
# ---------------------------------------------------------------------------
_tier2_available_cache: Optional[bool] = None


def tier2_available(local_url: Optional[str] = None,
                    local_model: Optional[str] = None) -> bool:
    """Check whether the local LLM (qwen3-1.7b via Ollama / LM Studio)
    is reachable.  Result is cached for the lifetime of the process.
    """
    global _tier2_available_cache
    if _tier2_available_cache is not None:
        return _tier2_available_cache

    try:
        from areval.providers.local_llm import LocalLLMProvider

        p = LocalLLMProvider(base_url=local_url, model=local_model)
        _tier2_available_cache = p.is_available()
    except Exception:
        _tier2_available_cache = False

    return _tier2_available_cache


def tier3_available() -> bool:
    """Check whether a remote LLM API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# TaskRouter
# ---------------------------------------------------------------------------
class TaskRouter:
    """Centralised per-task tier resolution.

    Replaces the per-Metric if-else chains with a single routing table.
    """

    def __init__(self, routing: Optional[dict[str, str]] = None) -> None:
        self._routing = routing or DEFAULT_TASK_ROUTING.copy()

    def resolve(self, task_name: str, provider: str = "auto") -> str:
        """Resolve which tier to use for *task_name*.

        Parameters
        ----------
        task_name : str
            Routing key (e.g. ``"faithfulness"``, ``"jailbreak_resistance"``).
        provider : str
            Override: ``"auto"`` (use routing table), ``"local"`` (force Tier 2),
            ``"llm"`` (force Tier 3), ``"mock"`` (force Tier 1).

        Returns
        -------
        str
            ``"tier1"``, ``"tier2"``, or ``"tier3"``.
        """
        # ---- explicit overrides — no fallback ----
        if provider == "mock":
            return "tier1"
        if provider == "local":
            if not tier2_available():
                raise RuntimeError(
                    "provider='local' but local LLM is not available"
                )
            return "tier2"
        if provider == "llm":
            if not tier3_available():
                raise RuntimeError(
                    "provider='llm' but no OPENAI_API_KEY is set"
                )
            return "tier3"

        # ---- auto: routing table + graceful degradation ----
        preferred = self._routing.get(task_name, "tier3")

        # Build fallback chain: preferred → alternative → mock
        if preferred == "tier1":
            return "tier1"

        chain = [preferred]
        chain.append("tier3" if preferred == "tier2" else "tier2")
        chain.append("tier1")

        for tier in chain:
            if self._available(tier):
                if tier != preferred:
                    log_degradation(
                        preferred.replace("tier", ""),
                        tier.replace("tier", ""),
                        f"{preferred} unavailable for task '{task_name}'",
                    )
                return tier

        return "tier1"

    def _available(self, tier: str) -> bool:
        if tier == "tier1":
            return True
        if tier == "tier2":
            return tier2_available()
        if tier == "tier3":
            return tier3_available()
        return False


# Global singleton — all Metrics share the same instance
router = TaskRouter()
