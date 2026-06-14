"""Unit tests for Judge implementations.

Currently focused on LLMJudge (Sprint 1.1). AgentJudge and DAGJudge tests
will be expanded in Sprint 1.3 / Sprint 1.5.
"""

import os

from areval.judges import DAGJudge, LLMJudge
from areval.test_case import AgentOutput, TestCase


class TestLLMJudge:
    """Tests for LLM-as-a-Judge."""

    def test_llm_judge_mock_scoring(self) -> None:
        judge = LLMJudge(provider="mock")
        test_case = TestCase(name="math", input="What is 2+2?")
        agent_output = AgentOutput(output="The answer is 4.")

        result = judge.evaluate(test_case, agent_output)

        assert 0.0 <= result.score <= 1.0
        assert result.passed == (result.score >= judge.threshold)
        assert result.reasoning
        assert result.threshold == 0.7

    def test_llm_judge_criteria_scores(self) -> None:
        judge = LLMJudge(provider="mock")
        test_case = TestCase(name="math", input="What is 2+2?")
        agent_output = AgentOutput(output="The answer is 4.")

        result = judge.evaluate(test_case, agent_output)

        expected_criteria = {"correctness", "completeness", "clarity", "helpfulness"}
        assert set(result.criteria_scores.keys()) == expected_criteria
        for score in result.criteria_scores.values():
            assert 0.0 <= score <= 1.0

    def test_llm_judge_empty_output(self) -> None:
        judge = LLMJudge(provider="mock")
        test_case = TestCase(name="math", input="What is 2+2?")
        agent_output = AgentOutput(output="")

        result = judge.evaluate(test_case, agent_output)

        assert 0.0 <= result.score <= 1.0
        assert result.reasoning

    def test_llm_judge_parse_response(self) -> None:
        judge = LLMJudge(provider="mock")
        response = judge._call_llm("dummy prompt")
        parsed = judge._parse_response(response)

        assert parsed["score"] == 0.75
        assert "correctness" in parsed["criteria_scores"]
        assert parsed["criteria_scores"]["correctness"] == 4.0 / 5.0

    def test_llm_judge_openai_no_key_fallback(self) -> None:
        """When no API key is present, openai provider should fall back to mock."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            judge = LLMJudge(provider="openai", model="gpt-4o-mini")
            test_case = TestCase(name="math", input="What is 2+2?")
            agent_output = AgentOutput(output="4")

            result = judge.evaluate(test_case, agent_output)

            assert 0.0 <= result.score <= 1.0
            assert result.reasoning
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key


class TestDAGJudge:
    """Minimal tests for DAG Judge (expanded in Sprint 1.5)."""

    def test_dag_judge_no_nodes(self) -> None:
        judge = DAGJudge()
        test_case = TestCase(name="t", input="hello")
        agent_output = AgentOutput(output="world")

        result = judge.evaluate(test_case, agent_output)

        assert result.score == 0.5
