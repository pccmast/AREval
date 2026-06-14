"""Unit tests for Judge implementations.

Sprint 1.1: LLMJudge mock/production paths.
Sprint 1.3: AgentJudge calculator tool, claim extraction, empty output.
Sprint 1.5: DAGJudge with nodes, expanded coverage.
"""

import os

from areval.judges import DAGJudge, LLMJudge, AgentJudge, JudgementNode, VerdictNode
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
        # Use evaluate() to provide expected/actual output for the heuristic
        tc = TestCase(
            name="parse_test",
            input="What is the capital of France?",
            expected_output="The capital of France is Paris.",
        )
        ao = AgentOutput(output="The capital of France is Paris.")
        result = judge.evaluate(tc, ao)

        assert result.score > 0.5  # heuristic should give high score for good match
        assert "correctness" in result.criteria_scores
        assert result.criteria_scores["correctness"] > 0.5

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


class TestAgentJudge:
    """Tests for Agent-as-a-Judge (Sprint 1.3 / 1.5)."""

    def test_agent_judge_extracts_claims(self) -> None:
        judge = AgentJudge()
        # Each sentence is >= 20 chars so should be kept as a claim
        claims = judge._extract_claims(
            "Python was created by Guido van Rossum. "
            "The first version was released in 1991. "
        )
        assert isinstance(claims, list)
        assert len(claims) >= 2

    def test_agent_judge_filters_short_claims(self) -> None:
        judge = AgentJudge()
        claims = judge._extract_claims("Hi. OK. This is a proper sentence with enough words to matter.")
        # "Hi" (2 chars) and "OK" (2 chars) should be filtered out (< 10 chars)
        assert len(claims) == 1
        assert "proper sentence" in claims[0]

    def test_agent_judge_extracts_claims_empty(self) -> None:
        judge = AgentJudge()
        claims = judge._extract_claims("")
        assert claims == []

    def test_agent_judge_calculator_tool(self) -> None:
        judge = AgentJudge()
        # "2 + 2 equals 4" should trigger the calculator
        result = judge._execute_tool("calculator", "2+2")
        assert result == "4.0"

    def test_agent_judge_calculator_complex(self) -> None:
        judge = AgentJudge()
        result = judge._execute_tool("calculator", "sqrt(16) + 2**3")
        # sqrt(16) = 4, 2**3 = 8 → 4 + 8 = 12
        assert result == "12.0"

    def test_agent_judge_calculator_error(self) -> None:
        judge = AgentJudge()
        result = judge._execute_tool("calculator", "1/0")
        assert result.startswith("Error:")

    def test_agent_judge_full_evaluation_math(self) -> None:
        judge = AgentJudge()
        test_case = TestCase(name="math", input="What is 2+2?")
        agent_output = AgentOutput(output="2 + 2 equals 4.")

        result = judge.evaluate(test_case, agent_output)

        assert 0.0 <= result.score <= 1.0
        assert "calculator" in result.reasoning.lower()
        assert result.metadata["claims_verified"] >= 1

    def test_agent_judge_full_evaluation_non_math(self) -> None:
        judge = AgentJudge()
        test_case = TestCase(
            name="history",
            input="Who wrote The Art of Computer Programming?",
        )
        agent_output = AgentOutput(
            output="Donald Knuth wrote The Art of Computer Programming, "
            "a classic multi-volume work on algorithms and their analysis."
        )

        result = judge.evaluate(test_case, agent_output)

        assert 0.0 <= result.score <= 1.0
        assert result.reasoning

    def test_agent_judge_empty_output(self) -> None:
        judge = AgentJudge()
        test_case = TestCase(name="empty", input="Say something.")
        agent_output = AgentOutput(output="")

        result = judge.evaluate(test_case, agent_output)

        assert 0.0 <= result.score <= 1.0
        assert result.metadata["claims_extracted"] == 0

    def test_agent_judge_search_tool_simulated(self) -> None:
        judge = AgentJudge()
        result = judge._execute_tool("search", "latest Python version")
        assert "Simulated" in result or "search" in result.lower()

    def test_agent_judge_unknown_tool(self) -> None:
        judge = AgentJudge()
        result = judge._execute_tool("nonexistent_tool", "query")
        assert "not found" in result.lower() or "Tool" in result


class TestDAGJudge:
    """Tests for DAG-based Judge (Sprint 1.5)."""

    def test_dag_judge_no_nodes(self) -> None:
        judge = DAGJudge()
        test_case = TestCase(name="t", input="hello")
        agent_output = AgentOutput(output="world")

        result = judge.evaluate(test_case, agent_output)

        assert result.score == 0.5

    def test_dag_judge_with_nodes(self) -> None:
        root = JudgementNode(
            criteria="Documentation is comprehensive",
            children=[
                VerdictNode(verdict=True, score=0.9),
                VerdictNode(verdict=False, score=0.1),
            ],
        )
        judge = DAGJudge(root_nodes=[root])
        test_case = TestCase(name="docs", input="Check docs")
        agent_output = AgentOutput(output="The documentation is comprehensive and clear.")

        result = judge.evaluate(test_case, agent_output)

        assert 0.0 <= result.score <= 1.0
        assert "node_scores" in result.metadata
