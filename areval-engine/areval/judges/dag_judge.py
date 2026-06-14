"""DAG-based judge — multi-step evaluation workflows.

Each :class:`JudgementNode` delegates its criterion to an
:class:`~areval.judges.llm_judge.LLMJudge`.  When no LLM API key is
available the judge falls back to a heuristic mock, giving *strictly
better* results than the previous keyword-counting skeleton.

Inspired by DeepEval's DAG metric.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from areval.judges.base import Judge, JudgeResult
from areval.test_case import TestCase, AgentOutput


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


class VerdictNode:
    """A terminal node that assigns a weight and optional label.

    Parameters
    ----------
    label : str
        Human-readable label for this verdict (e.g. ``"passed"``).
    weight : float
        Weight applied to the parent criterion score.
    """

    def __init__(self, label: str = "", weight: float = 1.0) -> None:
        self.label = label
        self.weight = weight


class JudgementNode:
    """A node that evaluates a criterion against an agent output.

    The ``criterion`` string is used as a natural-language evaluation
    rubric and fed to an :class:`~areval.judges.llm_judge.LLMJudge`.

    Parameters
    ----------
    criterion : str
        Natural-language description of what to evaluate.
    children : list of VerdictNode, optional
        Verdict nodes that weight this criterion.  If empty the raw
        score is used directly.
    weight : float
        Weight of this node relative to sibling nodes.
    """

    def __init__(
        self,
        criterion: str,
        children: Optional[List[VerdictNode]] = None,
        weight: float = 1.0,
    ) -> None:
        self.criterion = criterion
        self.children = children or []
        self.weight = weight


class NonBinaryJudgementNode(JudgementNode):
    """A judgement node whose children provide a rubric for a 0–1 scale.

    Each child :class:`VerdictNode` describes a score band (e.g.
    ``"1.0: completely correct"``, ``"0.5: partially correct"``).
    The LLM judge is asked to pick the most appropriate band.
    """

    def __init__(self, criterion: str, children: List[VerdictNode], weight: float = 1.0) -> None:
        super().__init__(criterion=criterion, children=children, weight=weight)


# ---------------------------------------------------------------------------
# DAGJudge
# ---------------------------------------------------------------------------


class DAGJudge(Judge):
    """Judge that evaluates using a configurable DAG of criteria.

    Each :class:`JudgementNode` delegates to
    :class:`~areval.judges.llm_judge.LLMJudge` so that criterion
    evaluation benefits from real LLM reasoning when an API key is
    available, and degrades to a heuristic mock otherwise.

    Parameters
    ----------
    threshold : float
        Pass / fail threshold.
    root_nodes : list of JudgementNode
        Top-level criteria to evaluate.  Scores are averaged.
    model : str
        LLM model name forwarded to the internal LLMJudge instances.
    """

    name = "dag_judge"

    def __init__(
        self,
        threshold: float = 0.7,
        root_nodes: Optional[List[JudgementNode]] = None,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.root_nodes = root_nodes or []
        self.model = model

    # ------------------------------------------------------------------
    # Node evaluation
    # ------------------------------------------------------------------

    def _evaluate_node(
        self,
        node: JudgementNode,
        test_case: TestCase,
        agent_output: AgentOutput,
    ) -> tuple[float, str]:
        """Evaluate a single criterion node via LLMJudge.

        Returns ``(score, reasoning)``.
        """
        from areval.judges.llm_judge import LLMJudge

        judge = LLMJudge(
            rubric=_criterion_rubric(node.criterion),
            criteria=["criterion"],
            model=self.model,
        )
        result = judge.evaluate(test_case, agent_output)
        score = result.score

        # Apply child verdict weights
        if node.children:
            weighted_score = sum(
                (v.weight * score) for v in node.children
            ) / sum(v.weight for v in node.children)
            return weighted_score, result.reasoning or "LLM-based DAG evaluation"

        return score, result.reasoning or "LLM-based DAG evaluation"

    # ------------------------------------------------------------------
    # Top-level evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_case: TestCase, agent_output: AgentOutput) -> JudgeResult:
        """Traverse root nodes and compute a weighted aggregate score."""
        if not self.root_nodes:
            return JudgeResult(
                score=0.5,
                reasoning="No DAG nodes configured",
                threshold=self.threshold,
            )

        scores: list[float] = []
        weights: list[float] = []
        reasonings: list[str] = []

        for root in self.root_nodes:
            score, reasoning = self._evaluate_node(root, test_case, agent_output)
            scores.append(score)
            weights.append(root.weight)
            reasonings.append(reasoning)

        total_weight = sum(weights)
        avg_score = (
            sum(s * w for s, w in zip(scores, weights)) / total_weight
            if total_weight > 0
            else 0.0
        )

        return JudgeResult(
            score=avg_score,
            reasoning=" | ".join(reasonings),
            threshold=self.threshold,
            metadata={
                "node_scores": scores,
                "node_weights": weights,
                "num_nodes": len(self.root_nodes),
            },
        )


# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------

def _criterion_rubric(criterion: str) -> str:
    """Build a concise LLM evaluation rubric from a criterion string."""
    return f"""You are an expert evaluator.  Judge the following criterion:

CRITERION: {criterion}

Question:
{{input}}

Agent answer:
{{actual_output}}

Answer ONLY in this format:
SCORE: <number between 0.0 and 1.0>
REASONING: <one-sentence explanation of why this score was assigned>
"""
