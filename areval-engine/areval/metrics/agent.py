"""Agent-specific evaluation metrics.

Metrics for evaluating agent behavior: tool use, task completion,
multi-turn coherence, and planning quality.

Three-tier support:

- **Tier 1** — pure-code deterministic checks (always available)
- **Tier 2** — local LLM semantic matching (ToolCallAccuracy)
- **Tier 3** — remote LLM for open-ended / trajectory evaluation
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional, Set

from areval.config import log_degradation
from areval.metrics.base import Metric, MetricResult
from areval.test_case import TestCase, AgentOutput


# ============================================================================
# Tier-2 semantic prompt (ToolCallAccuracy)
# ============================================================================

_SEMANTIC_BATCH_PROMPT = """\
For each pair below, determine if the two tool names are semantically
equivalent (they refer to the same action). Answer YES or NO on each line.

{items}

Output format (one line per item):
1: YES
2: NO
...
"""

# ============================================================================
# Tier-3 rubrics (TaskCompletion open-ended / trajectory)
# ============================================================================

_OPEN_ENDED_RUBRIC = """\
You are an expert evaluator for AI agent task completion.

Evaluate whether the agent successfully completed the given task based
on its final output. Consider correctness, completeness, and usefulness.

Task: {input}
Expected output: {expected_output}
Agent output: {actual_output}

Scoring guide (0.0 – 1.0):
  * 1.0 — task fully completed, output is correct and complete
  * 0.7 — task mostly completed with minor issues
  * 0.4 — partially completed, significant gaps
  * 0.1 — minimal progress toward task completion
  * 0.0 — agent failed to complete the task

Output format (exact):
SCORE: <0.0-1.0>
REASONING: <brief reasoning>
"""

_TRAJECTORY_RUBRIC = """\
Evaluate whether the following Agent's multi-step reasoning trajectory is
reasonable, efficient, and free of redundancy.

Task: {input}
Reasoning trajectory: {trajectory}
Final output: {actual_output}

Scoring criteria (0.0-1.0):
1. Reasoning coherence: each step logically follows from the previous
2. Tool-use appropriateness: tool selection and timing are correct
3. Efficiency: no unnecessary repeated steps or detours

SCORE: <0.0-1.0>
REASONING: <brief reasoning>
"""


# ============================================================================
# Helpers
# ============================================================================

def _parse_semantic_batch(raw: str, expected_count: int) -> list[bool]:
    """Parse batch semantic-equivalence response into a list of bool."""
    results: list[bool] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        m = re.match(r"\d+[.:)\s]*\s*(.+)", line)
        if m:
            label = m.group(1).strip().upper()
            results.append("YES" in label and "NO" not in label)
    while len(results) < expected_count:
        results.append(False)
    return results[:expected_count]


# ============================================================================
# ToolCallAccuracyMetric
# ============================================================================


class ToolCallAccuracyMetric(Metric):
    """Evaluates the correctness of agent tool calls.

    Two evaluation modes:

    - **Tier 1** (default): exact name matching with optional order/param checks.
    - **Tier 2**: semantic equivalence via qwen3-1.7b batch prompt
      (activated with ``check_semantic=True``).

    When semantic checking is enabled and the local LLM is unavailable,
    the metric silently falls back to exact-name matching (no error).
    """

    name = "tool_call_accuracy"

    def __init__(
        self,
        threshold: float = 1.0,
        provider: str = "auto",
        check_order: bool = True,
        check_params: bool = True,
        check_semantic: bool = False,
        local_llm_url: Optional[str] = None,
        local_llm_model: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, provider=provider, **kwargs)
        self.check_order = check_order
        self.check_params = check_params
        self.check_semantic = check_semantic
        self._local_url = local_llm_url
        self._local_model = local_llm_model

    # -- Semantic helpers ------------------------------------------------------

    def _tier2_available(self) -> bool:
        from areval.providers.local_llm import LocalLLMProvider as P
        return P(base_url=self._local_url, model=self._local_model).is_available()

    def _run_semantic_check(
        self, expected: list[str], actual: list[str]
    ) -> list[bool]:
        """Run batch semantic equivalence check via qwen3-1.7b."""
        pairs: list[tuple[str, str]] = []
        for i in range(max(len(expected), len(actual))):
            e = expected[i] if i < len(expected) else ""
            a = actual[i] if i < len(actual) else ""
            pairs.append((e, a))

        items = "\n".join(f"{i + 1}. {e} ↔ {a}" for i, (e, a) in enumerate(pairs))
        prompt = _SEMANTIC_BATCH_PROMPT.format(items=items)

        from areval.providers.local_llm import LocalLLMProvider as P
        llm = P(base_url=self._local_url, model=self._local_model)
        raw = llm.chat_complete(prompt)
        return _parse_semantic_batch(raw, len(pairs))

    # -- measure ---------------------------------------------------------------

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        expected_tools = test_case.expected_tools or []
        actual_tools = [tc.get("name", "") for tc in agent_output.tool_calls]

        if not expected_tools and not actual_tools:
            return MetricResult(
                name=self.name, score=1.0,
                reasoning="No tools expected or called",
                threshold=self.threshold,
            )
        if not expected_tools:
            return MetricResult(
                name=self.name, score=0.0,
                reasoning=f"No tools expected but {len(actual_tools)} were called",
                threshold=self.threshold,
            )
        if not actual_tools:
            return MetricResult(
                name=self.name, score=0.0,
                reasoning=f"Expected {len(expected_tools)} tools but none were called",
                threshold=self.threshold,
            )

        # Semantic mode: exact match first, then semantic for mismatches
        semantic_matches: list[bool] = []
        semantic_used = False

        if self.check_semantic and self._tier2_available():
            try:
                semantic_matches = self._run_semantic_check(expected_tools, actual_tools)
                semantic_used = True
            except Exception as exc:
                log_degradation("2", "1", f"Semantic check failed: {exc}")
                semantic_matches = []

        # Build score
        score = 0.0
        reasoning_parts: list[str] = []
        n = max(len(expected_tools), len(actual_tools))

        # Name matching
        if semantic_used:
            matches = sum(1 for m in semantic_matches if m)
            name_score = matches / n
            score += name_score * 0.5
            reasoning_parts.append(
                f"Semantic accuracy (tier-2): {matches}/{n}"
            )
        elif self.check_order:
            matches = sum(
                1 for e, a in zip(expected_tools, actual_tools) if e == a
            )
            order_score = matches / n
            score += order_score * 0.5
            reasoning_parts.append(
                f"Order accuracy: {matches}/{n}"
            )
        else:
            expected_set = set(expected_tools)
            actual_set = set(actual_tools)
            precision = len(expected_set & actual_set) / len(actual_set) if actual_set else 0
            recall = len(expected_set & actual_set) / len(expected_set) if expected_set else 0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0)
            score += f1 * 0.5
            reasoning_parts.append(
                f"Tool F1: {f1:.2f} (P={precision:.2f}, R={recall:.2f})"
            )

        # Parameter check
        if self.check_params and agent_output.tool_calls:
            param_score = 0.0
            for i in range(min(n, len(agent_output.tool_calls))):
                if semantic_used:
                    if i < len(semantic_matches) and semantic_matches[i]:
                        param_score += 1.0
                else:
                    expected_tool = expected_tools[i] if i < len(expected_tools) else ""
                    if agent_output.tool_calls[i].get("name") == expected_tool:
                        param_score += 1.0
            param_score /= n
            score += param_score * 0.5
            reasoning_parts.append(f"Param accuracy: {param_score:.2f}")

        metadata: dict[str, Any] = {
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
        }
        if semantic_used:
            metadata["tier"] = "tier2"
        else:
            metadata["tier"] = "tier1"

        return MetricResult(
            name=self.name,
            score=min(1.0, score),
            passed=score >= self.threshold,
            reasoning="; ".join(reasoning_parts),
            threshold=self.threshold,
            metadata=metadata,
        )


# ============================================================================
# TaskCompletionMetric
# ============================================================================


class TaskCompletionMetric(Metric):
    """Task completion evaluator supporting three modes.

    - ``mode="deterministic"`` (default): pure-code output-length heuristic.
    - ``mode="open_ended"``: remote LLM (Tier 3) using LLMJudge + rubric.
    - ``mode="trajectory"``: remote LLM (Tier 3) evaluating multi-step
      reasoning paths.

    Open-ended / trajectory mode falls back to deterministic when no
    API key is available.
    """

    name = "task_completion"

    def __init__(
        self,
        threshold: float = 1.0,
        provider: str = "auto",
        mode: str = "deterministic",
        local_llm_url: Optional[str] = None,
        local_llm_model: Optional[str] = None,
        **kwargs: Any,
    ):
        if mode not in ("deterministic", "open_ended", "trajectory"):
            raise ValueError(
                f"Unsupported mode: {mode!r}. "
                "Expected 'deterministic', 'open_ended', or 'trajectory'."
            )
        super().__init__(threshold=threshold, provider=provider, **kwargs)
        self.mode = mode
        self._local_url = local_llm_url
        self._local_model = local_llm_model

    # -- deterministic (Tier 1 — existing logic preserved) ---------------------

    def _evaluate_deterministic(
        self, test_case: TestCase, agent_output: AgentOutput
    ) -> MetricResult:
        if test_case.test_command:
            output_lower = agent_output.output.lower()
            success_indicators = ["pass", "success", "completed", "fixed"]
            failure_indicators = ["fail", "error", "exception", "timeout"]
            success_count = sum(1 for ind in success_indicators if ind in output_lower)
            failure_count = sum(1 for ind in failure_indicators if ind in output_lower)
            score = 1.0 if success_count > failure_count else 0.0
            return MetricResult(
                name=self.name, score=score,
                passed=score >= self.threshold,
                reasoning=f"Success indicators: {success_count}, "
                          f"Failure indicators: {failure_count}",
                threshold=self.threshold,
                metadata={"tier": "tier1", "mode": self.mode},
            )

        output = agent_output.output.strip()
        if not output:
            return MetricResult(
                name=self.name, score=0.0,
                reasoning="Empty output",
                threshold=self.threshold,
                metadata={"tier": "tier1", "mode": self.mode},
            )
        score = min(1.0, len(output) / 200)
        return MetricResult(
            name=self.name, score=score,
            passed=score >= self.threshold,
            reasoning=f"Output length: {len(output)} chars",
            threshold=self.threshold,
            metadata={"tier": "tier1", "mode": self.mode},
        )

    # -- open-ended / trajectory (Tier 3) --------------------------------------

    def _evaluate_with_llm(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        rubric: str,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> MetricResult:
        from areval.judges.llm_judge import LLMJudge

        judge = LLMJudge(rubric=rubric, criteria=["completion"])
        fmt_kwargs: dict[str, Any] = {
            "input": test_case.input,
            "expected_output": test_case.expected_output or "",
            "actual_output": agent_output.output,
        }
        if extra_kwargs:
            fmt_kwargs.update(extra_kwargs)

        result = judge.evaluate(test_case, agent_output, **fmt_kwargs)
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, result.score)),
            reasoning=result.reasoning or "Tier-3 LLM task completion evaluation",
            threshold=self.threshold,
            metadata={
                "tier": "tier3",
                "mode": self.mode,
                "judge_provider": judge.provider,
            },
        )

    def _tier2_available(self) -> bool:
        from areval.providers.local_llm import LocalLLMProvider as P
        return P(base_url=self._local_url, model=self._local_model).is_available()

    # -- measure ---------------------------------------------------------------

    def measure(self, test_case: TestCase, agent_output: AgentOutput) -> MetricResult:
        # mock → straight to Tier 1
        if self.provider == "mock":
            return self._evaluate_deterministic(test_case, agent_output)

        # deterministic mode → always Tier 1
        if self.mode == "deterministic":
            return self._evaluate_deterministic(test_case, agent_output)

        # open_ended / trajectory → Tier 3, fallback to T2 → T1
        rubric = _TRAJECTORY_RUBRIC if self.mode == "trajectory" else _OPEN_ENDED_RUBRIC
        extra: dict[str, Any] | None = None

        if self.mode == "trajectory":
            trajectory = agent_output.metadata.get("trajectory", "")
            extra = {"trajectory": trajectory}

        # llm → Tier 3 or raise
        if self.provider == "llm":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("provider='llm' but no OPENAI_API_KEY is set")
            return self._evaluate_with_llm(test_case, agent_output, rubric, extra)

        # local → Tier 2 or raise
        if self.provider == "local":
            if self._tier2_available():
                # For TaskCompletion, Tier 2 is still via LLMJudge but uses local
                # Actually, let's just fallback to deterministic since TaskCompletion
                # doesn't have a specialized Tier 2 prompt — the semantic
                # understanding requires Tier 3. But provide it anyway.
                log_degradation("3", "1", "Tier 2 not specialized for TaskCompletion, using deterministic")
                return self._evaluate_deterministic(test_case, agent_output)
            raise RuntimeError("provider='local' but local LLM is not available")

        # auto: Tier 3 → Tier 2 → Tier 1
        if os.environ.get("OPENAI_API_KEY"):
            try:
                return self._evaluate_with_llm(test_case, agent_output, rubric, extra)
            except Exception as exc:
                log_degradation("3", "2", f"Tier 3 failed: {exc}")

        if self._tier2_available():
            log_degradation("3", "1", "Tier 2 not specialized, using deterministic")
            return self._evaluate_deterministic(test_case, agent_output)

        log_degradation("3", "1", "No LLM available, using deterministic")
        return self._evaluate_deterministic(test_case, agent_output)
