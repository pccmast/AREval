"""DAG-based judge inspired by DeepEval's DAG metric.

Allows complex, multi-step evaluation workflows using a directed
acyclic graph of judgement nodes.
"""

from typing import Any, Dict, List, Optional, Union

from areval.judges.base import Judge, JudgeResult
from areval.test_case import TestCase, AgentOutput


class VerdictNode:
    """A leaf node that assigns a score."""

    def __init__(
        self,
        verdict: Union[str, bool],
        score: float = 1.0,
        child: Optional["JudgementNode"] = None,
    ):
        self.verdict = verdict
        self.score = score
        self.child = child


class JudgementNode:
    """A node in the evaluation DAG.

    Can be a binary judgement (yes/no), non-binary (scale),
    or task node (extracts information).
    """

    def __init__(
        self,
        criteria: str,
        children: Optional[List[VerdictNode]] = None,
        evaluation_params: Optional[List[str]] = None,
    ):
        self.criteria = criteria
        self.children = children or []
        self.evaluation_params = evaluation_params or []


class DAGJudge(Judge):
    """Judge that evaluates using a configurable DAG workflow.

    Example DAG for code review evaluation:

    1. Extract code blocks (TaskNode)
    2. Check if code compiles (BinaryJudgementNode)
       - Yes → Check test coverage (NonBinaryJudgementNode)
       - No → Score 0
    3. Check documentation (BinaryJudgementNode)
    4. Aggregate scores

    Inspired by DeepEval's DAGMetric.
    """

    name = "dag_judge"

    def __init__(
        self,
        threshold: float = 0.7,
        root_nodes: Optional[List[JudgementNode]] = None,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.root_nodes = root_nodes or []

    def _evaluate_node(
        self,
        node: JudgementNode,
        test_case: TestCase,
        agent_output: AgentOutput,
    ) -> tuple[float, str]:
        """Evaluate a single node and return (score, reasoning)."""
        # In production: Use LLM to evaluate the criteria
        # For skeleton: Use heuristic scoring

        # Check criteria against output
        criteria_lower = node.criteria.lower()
        output_lower = agent_output.output.lower()

        # Simple keyword matching as proxy
        keywords = [w for w in criteria_lower.split() if len(w) > 4]
        matches = sum(1 for kw in keywords if kw in output_lower)
        match_ratio = matches / len(keywords) if keywords else 0.5

        # Find matching verdict
        best_score = 0.0
        best_verdict = "no_match"

        for child in node.children:
            if isinstance(child.verdict, bool):
                if match_ratio > 0.5 and child.verdict:
                    best_score = child.score
                    best_verdict = "yes"
                    # Follow child chain
                    if child.child:
                        child_score, child_reason = self._evaluate_node(
                            child.child, test_case, agent_output
                        )
                        best_score = (best_score + child_score) / 2
                elif match_ratio <= 0.5 and not child.verdict:
                    best_score = child.score
                    best_verdict = "no"
            elif isinstance(child.verdict, str):
                # String verdict matching
                if child.verdict.lower() in ["any", "default"]:
                    best_score = max(best_score, child.score)
                    best_verdict = child.verdict

        return best_score, f"Criteria '{node.criteria[:50]}...': {best_verdict}"

    def evaluate(self, test_case: TestCase, agent_output: AgentOutput) -> JudgeResult:
        """Traverse the DAG and compute aggregate score."""
        if not self.root_nodes:
            return JudgeResult(
                score=0.5,
                reasoning="No DAG nodes configured",
                threshold=self.threshold,
            )

        scores = []
        reasonings = []

        for root in self.root_nodes:
            score, reasoning = self._evaluate_node(root, test_case, agent_output)
            scores.append(score)
            reasonings.append(reasoning)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return JudgeResult(
            score=avg_score,
            reasoning=" | ".join(reasonings),
            threshold=self.threshold,
            metadata={
                "node_scores": scores,
                "num_nodes": len(self.root_nodes),
            },
        )
