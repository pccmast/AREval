"""Judge implementations for LLM-as-a-Judge and Agent-as-a-Judge patterns.

All built-in judges are auto-registered so CLI YAML configs can reference
them by name via ``get_judge(name, **config)``.
"""

from areval.judges.base import Judge, JudgeResult, get_judge, register_judge
from areval.judges.llm_judge import LLMJudge
from areval.judges.agent_judge import AgentJudge
from areval.judges.dag_judge import (
    DAGJudge,
    JudgementNode,
    VerdictNode,
    NonBinaryJudgementNode,
)

# ---------------------------------------------------------------------------
# Auto-register built-in judges
# ---------------------------------------------------------------------------
register_judge("llm_judge", LLMJudge)
register_judge("agent_judge", AgentJudge)
register_judge("dag_judge", DAGJudge)

__all__ = [
    "Judge",
    "JudgeResult",
    "get_judge",
    "register_judge",
    "LLMJudge",
    "AgentJudge",
    "DAGJudge",
    "JudgementNode",
    "VerdictNode",
    "NonBinaryJudgementNode",
]
