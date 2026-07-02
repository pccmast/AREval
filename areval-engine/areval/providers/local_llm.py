"""Local LLM provider via OpenAI-compatible HTTP API.

Supports LM Studio (default), Ollama, and any other backend that exposes
an OpenAI-compatible ``/v1/chat/completions`` endpoint.

Environment variables
---------------------
``LM_API_TOKEN``
    API token for LM Studio (required for LM Studio >= 0.3.x).
    If set, passed as ``Authorization: Bearer <token>``.
    If not set, falls back to ``"not-needed"`` (works for Ollama, vLLM, older LM Studio).

``AREVAL_LOCAL_LLM_URL``
    Base URL override.  Default ``http://localhost:12345/v1``.

``AREVAL_LOCAL_LLM_MODEL``
    Model name override.  Default ``qwen3-1.7b``.

Usage::

    from areval.providers.local_llm import LocalLLMProvider

    provider = LocalLLMProvider()
    if provider.is_available():
        reply = provider.chat_complete("What is the capital of France?")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from areval.providers.base import LocalModelProvider

# Auto-load .env from project root (no third-party deps)
for _env_dir in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
    _env_file = _env_dir / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                if _k not in os.environ:
                    os.environ[_k] = _v
        break

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LOCAL_LLM_URL = os.environ.get("AREVAL_LOCAL_LLM_URL", "http://localhost:12345/v1")
DEFAULT_MODEL = os.environ.get("AREVAL_LOCAL_LLM_MODEL", "qwen3-1.7b")
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Cached availability probe result (module-level)
# ---------------------------------------------------------------------------
_availability_cache: Optional[bool] = None
_cached_base_url: Optional[str] = None


def _probe_http(url: str, timeout: float = 0.5, api_key: Optional[str] = None) -> bool:
    """Lightweight HTTP reachability check.

    Returns
    -------
    bool
        ``True`` if the endpoint responds with any non-5xx status.
    """
    try:
        import httpx

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = httpx.get(url, timeout=timeout, follow_redirects=False, headers=headers)
        return r.status_code < 500  # 4xx counts as "reachable"
    except (
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    ):
        return False
    except Exception:
        return False


class LocalLLMProvider(LocalModelProvider):
    """OpenAI-compatible local LLM backend (LM Studio / Ollama / vLLM).

    Parameters
    ----------
    base_url : str, optional
        Base URL of the OpenAI-compatible API endpoint.
        Defaults to ``http://localhost:12345/v1`` (LM Studio default).
    model : str, optional
        Model name to pass in chat completion requests.
        Defaults to ``"qwen3-1.7b"``.
    timeout : float
        HTTP request timeout in seconds.  Default ``30.0``.
    max_retries : int
        Number of retry attempts.  Default ``2``.
    api_key : str, optional
        API key if required (most local servers accept any non-empty string).
        Defaults to ``"not-needed"`` for local backends.
    """

    name = "local_llm"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        api_key: Optional[str] = None,
    ) -> None:
        self._base_url = (base_url or DEFAULT_LOCAL_LLM_URL).rstrip("/")
        self._model = model or DEFAULT_MODEL
        self._timeout = timeout
        self._max_retries = max_retries
        # LM Studio >= 0.3.x requires an API token; Ollama/vLLM accept any non-empty string
        self._api_key = api_key or os.environ.get("LM_API_TOKEN") or "not-needed"

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------
    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    # ------------------------------------------------------------------
    #  Availability
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Probe the models endpoint; cache result per URL.

        Only re-probes when *base_url* changes (e.g. user switches profile).
        """
        global _availability_cache, _cached_base_url

        if _availability_cache is not None and _cached_base_url == self._base_url:
            return _availability_cache

        models_url = f"{self._base_url}/models"
        ok = _probe_http(models_url, timeout=0.5, api_key=self._api_key)

        _availability_cache = ok
        _cached_base_url = self._base_url
        return ok

    # ------------------------------------------------------------------
    #  Chat completion
    # ------------------------------------------------------------------
    def chat_complete(self, prompt: str, **kwargs: str) -> str:
        """Send a prompt and return the raw response text.

        Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint.
        Disables reasoning/thinking mode via ``extra_body`` — evaluation
        needs the final answer, not the chain-of-thought trace.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LocalLLMProvider. "
                "Install with: pip install openai"
            ) from exc

        client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body={"enable_thinking": False},  # 禁用 Qwen3 等模型的推理模式
        )

        # 处理思考模式模型: 取非空的 message.content，
        # reasoning_content 在思考模式下可能占据 choices[0] 而 content 为 None
        for choice in response.choices:
            content = choice.message.content
            if content:
                return content

        return ""

    def __repr__(self) -> str:
        return f"LocalLLMProvider(base_url={self._base_url!r}, " f"model={self._model!r})"
