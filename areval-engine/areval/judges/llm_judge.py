"""LLM-as-a-Judge implementation.

Uses a powerful LLM to evaluate agent outputs against rubrics.
Inspired by DeepEval's G-Eval and industry best practices.

Supported providers:
- "openai": calls OpenAI Chat Completions API
- "anthropic": calls Anthropic Messages API
- "mock": returns a heuristic simulated response (for CI / offline use,
  produces distinguishable scores rather than a fixed constant)
"""

import os
from typing import Any, Dict, List, Optional

from areval.judges.base import Judge, JudgeResult
from areval.test_case import AgentOutput, TestCase


class LLMJudge(Judge):
    """Judge that uses an LLM to evaluate outputs.

    Supports custom rubrics, multi-criteria scoring, and
    chain-of-thought reasoning extraction.
    """

    name = "llm_judge"

    # Default evaluation rubric template
    DEFAULT_RUBRIC = """
You are an expert evaluator assessing AI agent responses.

Evaluate the following response based on these criteria (score 1-5 each):

1. Correctness: Is the information accurate and factually correct?
2. Completeness: Does it fully address all aspects of the question?
3. Clarity: Is the response well-structured and easy to understand?
4. Helpfulness: Does it provide actionable, useful information?

Question: {input}
Expected Answer: {expected_output}
Agent Response: {actual_output}

Provide your evaluation in this exact format:
SCORE: <overall_score_0_to_1>
CORRECTNESS: <1-5>
COMPLETENESS: <1-5>
CLARITY: <1-5>
HELPFULNESS: <1-5>
REASONING: <your detailed reasoning>
"""

    def __init__(
        self,
        threshold: float = 0.7,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        rubric: Optional[str] = None,
        criteria: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.rubric = rubric or self.DEFAULT_RUBRIC
        self.criteria = criteria or ["correctness", "completeness", "clarity", "helpfulness"]

    def _api_key_from_env(self) -> Optional[str]:
        """Read API key from environment variables."""
        if self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        if self.provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        return None

    def _mock_response(
        self,
        expected_output: str = "",
        actual_output: str = "",
        context: str = "",
        **kwargs: Any,  # accept extra rubric fields silently
    ) -> str:
        """Heuristic simulated LLM response for offline/CI usage.

        Produces a distinguishable score based on token-level overlap
        between a *reference text* and the actual output.

        Reference precedence:
        1. ``context``     – used by RAG metrics (faithfulness / precision)
        2. ``expected_output`` – used by most other metrics
        3. If neither is non-empty, returns a neutral 0.50.
        """
        ref_text = context or expected_output

        if not ref_text or not actual_output:
            return """
SCORE: 0.50
CORRECTNESS: 3
COMPLETENESS: 2
CLARITY: 3
HELPFULNESS: 2
REASONING: Unable to compute heuristic — missing reference or actual output.
"""

        ref_tokens = set(ref_text.lower().split())
        act_tokens = set(actual_output.lower().split())

        if not ref_tokens or not act_tokens:
            overlap = 0.0
        else:
            overlap = len(ref_tokens & act_tokens) / len(ref_tokens | act_tokens)

        # Map Jaccard [0, 1] → score [0.30, 0.95]
        score = 0.30 + overlap * 0.65

        # Mild length penalty: reward responses within 50%–200% of expected length
        len_ratio = len(actual_output) / max(len(ref_text), 1)
        if len_ratio < 0.5 or len_ratio > 2.0:
            score = max(0.25, score - 0.10)

        score = round(score, 2)

        # Map score to 1-5 sub-scores proportionally
        sub_score = max(1, min(5, round(score * 5)))
        completeness_sub = max(1, min(5, round(overlap * 5)))

        return f"""
SCORE: {score}
CORRECTNESS: {sub_score}
COMPLETENESS: {completeness_sub}
CLARITY: {sub_score}
HELPFULNESS: {sub_score}
REASONING: Heuristic evaluation — token overlap={overlap:.2f}, length_ratio={len_ratio:.2f}
"""

    def _call_openai(self, prompt: str, api_key: str) -> str:
        """Call OpenAI Chat Completions API."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for provider='openai'. "
                "Install with: pip install openai"
            ) from e

        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=3)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str, api_key: str) -> str:
        """Call Anthropic Messages API."""
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for provider='anthropic'. "
                "Install with: pip install anthropic"
            ) from e

        client = Anthropic(api_key=api_key, timeout=60.0, max_retries=3)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        if response.content:
            return getattr(response.content[0], "text", "")
        return ""

    def _call_llm(self, prompt: str, **format_kwargs: Any) -> str:
        """Call the LLM with the evaluation prompt.

        Falls back to the mock response when:
        - provider is explicitly "mock"
        - no API key is configured and provider is openai/anthropic

        ``format_kwargs`` is forwarded to :meth:`_mock_response` so that
        the heuristic can pick the best reference text (e.g. ``{context}``
        for RAG metrics, ``{expected_output}`` otherwise).
        """
        if self.provider == "mock":
            return self._mock_response(**format_kwargs)

        api_key = self.api_key or self._api_key_from_env()
        if not api_key:
            print(
                f"[LLMJudge] Warning: no API key found for provider={self.provider!r}. "
                "Falling back to heuristic mock response."
            )
            return self._mock_response(**format_kwargs)

        if self.provider == "openai":
            return self._call_openai(prompt, api_key)
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, api_key)

        raise ValueError(f"Unsupported LLM provider: {self.provider!r}")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse structured LLM response."""
        result: Dict[str, Any] = {
            "score": 0.5,
            "reasoning": "",
            "criteria_scores": {},
        }

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    result["score"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()
                if key in [c.lower() for c in self.criteria]:
                    try:
                        result["criteria_scores"][key] = float(val.strip()) / 5.0
                    except ValueError:
                        pass

        return result

    def evaluate(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        **extra_format_kwargs: Any,
    ) -> JudgeResult:
        """Run LLM evaluation.

        Parameters
        ----------
        test_case : TestCase
        agent_output : AgentOutput
        extra_format_kwargs :
            Additional keyword arguments passed to :meth:`str.format` on the
            rubric template.  Useful for rubrics that require extra fields
            such as ``{context}``.
        """
        expected = test_case.expected_output or ""
        actual = agent_output.output
        format_kwargs: Dict[str, Any] = {
            "input": test_case.input,
            "expected_output": expected,
            "actual_output": actual,
        }
        format_kwargs.update(extra_format_kwargs)
        prompt = self.rubric.format(**format_kwargs)

        response = self._call_llm(prompt, **format_kwargs)
        parsed = self._parse_response(response)

        return JudgeResult(
            score=parsed["score"],
            reasoning=parsed["reasoning"],
            criteria_scores=parsed.get("criteria_scores", {}),
            threshold=self.threshold,
            metadata={
                "model": self.model,
                "provider": self.provider,
            },
        )
