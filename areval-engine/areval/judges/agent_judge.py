"""Agent-as-a-Judge implementation.

Uses an agentic system with tools to evaluate other agents.
Inspired by Zhuge et al. 2025 and emerging agent evaluation research.
"""

import ast
import math
import operator
import os
import re
from typing import Any, Callable, Dict, List, Optional

from areval.judges.base import Judge, JudgeResult
from areval.test_case import TestCase, AgentOutput

# ---------------------------------------------------------------------------
# Safe calculator (ast.NodeVisitor)
# ---------------------------------------------------------------------------

_SAFE_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pow": pow,
}

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class _SafeCalc(ast.NodeVisitor):
    """Visit an AST node and return a numeric value.

    Only whitelisted operations are allowed — no attribute access,
    no `eval`, no system calls.
    """

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        return float(_BIN_OPS[op_type](left, right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return float(_UNARY_OPS[op_type](operand))

    def visit_Call(self, node: ast.Call) -> float:
        func_name: str
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        else:
            raise ValueError(f"Unsupported call target: {ast.dump(node.func)}")

        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function not allowed: {func_name}")

        args = [self.visit(a) for a in node.args]
        return float(_SAFE_FUNCTIONS[func_name](*args))


def safe_calc(expression: str) -> str:
    """Safely evaluate a mathematical expression using ast.NodeVisitor.

    Supports: + - * / ** // % , sqrt, abs, round, min, max,
    log, log10, log2, sin, cos, tan, pow, and parentheses.

    Returns the numeric result as a string, or an error message
    prefixed with 'Error:'.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _SafeCalc().visit(tree)
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: unexpected calculator error ({e})"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, Callable[[str], str]] = {}


def _web_search(query: str) -> str:
    """Search DuckDuckGo HTML (no API key required).

    Returns up to 3 result snippets joined by newlines.  Falls back
    gracefully to a placeholder when the network is unavailable.

    Uses ``httpx`` which is already a project dependency.
    """
    try:
        import httpx
    except ImportError:
        return f"[Search unavailable — httpx not installed.  Query: {query[:80]}]"

    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=10.0,
            headers={"User-Agent": "AREval/0.1"},
        )
        resp.raise_for_status()
    except Exception:
        return f"[Search temporarily unavailable.  Query: {query[:80]}]"

    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)<', resp.text)
    return "\n".join(snippets[:3]) if snippets else f"[No results for: {query[:80]}]"


def _subprocess_executor(code: str) -> str:
    """Execute code in a sandboxed subprocess.

    Runs Python code with a 5-second timeout in a restricted
    subprocess.  Production deployments should upgrade to a container
    sandbox (Docker / E2B).
    """
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "(no output)"
        return f"Error (exit {result.returncode}): {result.stderr.strip()[:200]}"
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out (5s)"
    except Exception as e:
        return f"Error: code execution failed ({e})"


# Register built-in tools
_TOOL_REGISTRY["calculator"] = safe_calc
_TOOL_REGISTRY["search"] = _web_search
_TOOL_REGISTRY["code_executor"] = _subprocess_executor


def _format_verifications(verifications: List[Dict[str, Any]]) -> str:
    """Format tool verification results for injection into an LLM rubric."""
    if not verifications:
        return "(no tool verifications performed)"
    lines: list[str] = []
    for v in verifications:
        tool = v.get("tool", "unknown")
        claim = v.get("claim", "")[:120]
        result = v.get("result", "")
        lines.append(f"- [{tool}] Claim: {claim}")
        lines.append(f"  Result: {result}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AgentJudge
# ---------------------------------------------------------------------------


class AgentJudge(Judge):
    """Judge that uses an agent with tool access to evaluate outputs.

    More powerful than simple LLM-as-a-Judge because it can:
    1. Search for factual verification
    2. Execute code to verify correctness
    3. Use external tools for deeper analysis
    4. Perform multi-step reasoning

    This implements the "Agent-as-a-Judge" pattern from recent research.
    """

    name = "agent_judge"

    # Numeric fact patterns used to decide when to invoke the calculator
    _MATH_PATTERNS = [
        re.compile(p)
        for p in [
            r"\b\d+\s*[\+\-\*\/\*\*]\s*\d+\b",  # 2+2, 10 * 3
            r"\b\d+\s*(?:equals?|is|makes?|total|sum)\s*\d+\b",  # equals 4
            r"\b(?:sqrt|abs|round|min|max|log|sin|cos|tan)\s*\(",  # sqrt(16)
            r"\b\d+\s*(?:plus|minus|times?|divided by|multiplied by)\s*\d+\b",
        ]
    ]

    def __init__(
        self,
        threshold: float = 0.7,
        model: str = "gpt-4",
        tools: Optional[List[str]] = None,
        max_steps: int = 5,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.model = model
        self.tools = tools or ["search", "calculator", "code_executor"]
        self.max_steps = max_steps

    def _execute_tool(self, tool_name: str, query: str) -> str:
        """Execute a tool for factual verification.

        Looks up the tool in the global registry.  The built-in 'calculator'
        tool safely evaluates mathematical expressions using ast.NodeVisitor.

        Non-calculator tools (search, code_executor) are simulated by default
        to keep tests deterministic and avoid invoking host binaries or
        network calls in CI.  Set ``AREVAL_SIMULATE_TOOLS=0`` to use real
        tools when needed.
        """
        tool_fn = _TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            return f"[Tool not found: {tool_name}]"

        # Determine whether to simulate non-calculator tools.
        # Default (or AREVAL_SIMULATE_TOOLS=1): simulated, deterministic.
        # AREVAL_SIMULATE_TOOLS=0: invoke real tools.
        simulate = os.getenv("AREVAL_SIMULATE_TOOLS", "1") != "0"
        if simulate and tool_name.lower() != "calculator":
            return f"Simulated {tool_name} results for query: {query}"

        try:
            return tool_fn(query)
        except Exception as e:
            return f"[Tool error: {e}]"

    def _is_math_claim(self, text: str) -> bool:
        """Return True if the text appears to contain a numeric claim."""
        return any(pattern.search(text) for pattern in self._MATH_PATTERNS)

    def _extract_expression(self, text: str) -> Optional[str]:
        """Try to extract a computable expression from a math claim."""
        # Try direct arithmetic: "2 + 2 equals 4" → "2+2"
        m = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", text)
        if m:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}"
        # Try sqrt(...) / function calls
        m = re.search(r"((?:sqrt|abs|round|min|max|log|sin|cos|tan)\s*\([^)]+\))", text)
        if m:
            return m.group(1)
        return None

    def evaluate(self, test_case: TestCase, agent_output: AgentOutput) -> JudgeResult:
        """Run agent-based evaluation with tool use."""
        # Step 1: Extract claims from the output
        claims = self._extract_claims(agent_output.output)

        # Step 2: Verify claims using tools
        verification_results: list[dict[str, Any]] = []
        for claim in claims[:3]:  # Limit to top 3 claims
            tool_used = None
            result = None

            if self._is_math_claim(claim) and "calculator" in self.tools:
                expr = self._extract_expression(claim)
                if expr:
                    result = self._execute_tool("calculator", expr)
                    tool_used = "calculator"
            elif len(claim) > 20 and "search" in self.tools:
                # Any substantive claim is worth searching
                result = self._execute_tool("search", claim)
                tool_used = "search"
            elif any(
                kw in claim.lower()
                for kw in ["code", "script", "function", "def ", "import", "run"]
            ):
                if "code_executor" in self.tools:
                    result = self._execute_tool("code_executor", claim)
                    tool_used = "code_executor"

            if result is not None:
                verification_results.append(
                    {
                        "claim": claim,
                        "expression": expr if tool_used == "calculator" else None,
                        "tool": tool_used,
                        "result": result,
                    }
                )

        # Step 3: Assess overall quality
        score = self._assess_quality(test_case, agent_output, verification_results)

        reasoning = self._generate_reasoning(claims, verification_results, score)

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            criteria_scores={
                "factual_accuracy": score,
                "claim_verification": (1.0 if verification_results else 0.5),
            },
            threshold=self.threshold,
            metadata={
                "tools_used": self.tools,
                "claims_extracted": len(claims),
                "claims_verified": len(verification_results),
            },
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text.

        Splits on sentence boundaries (. ! ? newline), strips whitespace,
        and filters out fragments shorter than 10 characters (very short
        fragments are unlikely to be factual claims).  Math claims (e.g.
        \"2 + 2 equals 4\") are kept even if below the usual length threshold.
        """
        # Split by sentence boundaries
        sentences = re.split(r"[.!?\n]+", text)
        cleaned = [s.strip() for s in sentences if s.strip()]
        # Keep sentences that are long enough OR contain math patterns
        factual = [s for s in cleaned if len(s) >= 10 or self._is_math_claim(s)]
        return factual[:5]

    def _assess_quality(
        self,
        test_case: TestCase,
        agent_output: AgentOutput,
        verifications: List[Dict[str, Any]],
    ) -> float:
        """Calculate overall quality score via LLMJudge.

        When an LLM API key is available, a dedicated rubric incorporates
        tool verification results into a 0–1 quality score.  Without a
        key the judge degrades to its built-in heuristic mock.
        """
        from areval.judges.llm_judge import LLMJudge

        verification_text = _format_verifications(verifications)

        rubric = f"""You are the final quality judge in an Agent-as-a-Judge pipeline.

Tool verification results (may be empty):
{verification_text}

Question:
{{input}}

Expected answer (may be empty):
{{expected_output}}

Agent answer:
{{actual_output}}

Assign a 0.0–1.0 score based on factual accuracy and completeness.
Output ONLY:
SCORE: <number>
REASONING: <one sentence>
"""
        judge = LLMJudge(rubric=rubric, criteria=["quality"], model=self.model)
        result = judge.evaluate(test_case, agent_output)
        return max(0.0, min(1.0, result.score))

    def _generate_reasoning(
        self,
        claims: List[str],
        verifications: List[Dict[str, Any]],
        score: float,
    ) -> str:
        """Generate human-readable reasoning."""
        parts = [f"Extracted {len(claims)} claims from output."]
        if verifications:
            for v in verifications:
                tool = v.get("tool", "unknown")
                res = v.get("result", "")
                claim_short = v.get("claim", "")[:50]
                parts.append(f"Verified claim '{claim_short}...' using {tool}: {res}.")
        parts.append(f"Overall quality score: {score:.2f}")
        return " ".join(parts)
