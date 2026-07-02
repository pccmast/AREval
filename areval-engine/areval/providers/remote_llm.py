"""Remote LLM provider – OpenAI, Anthropic, and custom endpoints.

Usage::

    from areval.providers.remote_llm import RemoteLLMProvider

    provider = RemoteLLMProvider(provider="openai", api_key="sk-...")
    if provider.is_available():
        reply = provider.chat_complete("Evaluate ...")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from areval.providers.base import RemoteLLMProvider as _BaseRemoteLLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 60.0       # Matches existing LLMJudge default
DEFAULT_MAX_RETRIES = 3      # Matches existing LLMJudge default


class RemoteLLMProvider(_BaseRemoteLLMProvider):
    """Unified remote LLM API client (OpenAI / Anthropic / custom).

    **Design constraint (ADR-1 / 4.2)**:
    - This provider ONLY handles HTTP calls: prompt → raw text.
    - It does NOT do key resolution, mock fallback, or response parsing.
    - Key resolution stays in ``LLMJudge._call_llm()``.
    - Mock path stays in ``LLMJudge._mock_response()``.

    Parameters
    ----------
    provider : str
        Backend identifier: ``"openai"``, ``"anthropic"``, or ``"custom"``.
    api_key : str, optional
        API key.  If ``None``, reads from environment:
        - ``OPENAI_API_KEY`` for provider="openai"
        - ``ANTHROPIC_API_KEY`` for provider="anthropic"
        - ``AREVAL_REMOTE_API_KEY`` for provider="custom"
    model : str
        Model name.  Default ``"gpt-4o-mini"``.
    base_url : str, optional
        Base URL override for custom / compatible endpoints.
    timeout : float
        HTTP timeout in seconds.  Default ``60.0``.
    max_retries : int
        Retry count.  Default ``3``.
    """

    name = "remote_llm"

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if provider not in ("openai", "anthropic", "custom"):
            raise ValueError(
                f"Unsupported provider: {provider!r}. "
                f"Expected 'openai', 'anthropic', or 'custom'."
            )
        self._provider = provider
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries

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
    def provider_name(self) -> str:
        return self._provider

    # ------------------------------------------------------------------
    #  Key resolution (stays here for is_available; LLMJudge has its own)
    # ------------------------------------------------------------------
    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API key: explicit > env var."""
        if self._api_key:
            return self._api_key
        if self._provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        if self._provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        if self._provider == "custom":
            return os.environ.get("AREVAL_REMOTE_API_KEY")
        return None

    # ------------------------------------------------------------------
    #  Availability
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Check API key + lightweight endpoint probe."""
        key = self._resolve_api_key()
        if not key:
            return False

        target_url = self._base_url
        if not target_url:
            if self._provider == "openai":
                target_url = os.environ.get(
                    "AREVAL_LLM_BASE_URL", "https://api.openai.com/v1"
                )
            elif self._provider == "anthropic":
                target_url = "https://api.anthropic.com"
            else:
                return False

        return self._probe_http(f"{target_url.rstrip('/')}/models", timeout=2.0)

    @staticmethod
    def _probe_http(url: str, timeout: float = 2.0) -> bool:
        """Lightweight reachability check."""
        try:
            import httpx

            r = httpx.get(url, timeout=timeout, follow_redirects=False)
            return r.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError,
                httpx.RemoteProtocolError):
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    #  Chat completion – OpenAI path
    # ------------------------------------------------------------------
    def _chat_openai(self, prompt: str, api_key: str) -> str:
        """Call OpenAI Chat Completions."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for provider='openai'. "
                "Install with: pip install openai"
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    # ------------------------------------------------------------------
    #  Chat completion – Anthropic path
    # ------------------------------------------------------------------
    def _chat_anthropic(self, prompt: str, api_key: str) -> str:
        """Call Anthropic Messages API."""
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for provider='anthropic'. "
                "Install with: pip install anthropic"
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        client = Anthropic(**client_kwargs)
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        if response.content:
            return getattr(response.content[0], "text", "")
        return ""

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def chat_complete(self, prompt: str) -> str:
        """Run chat completion and return raw response text.

        Parameters
        ----------
        prompt : str
            The full prompt to send.

        Returns
        -------
        str
            Raw model response.

        Raises
        ------
        ValueError
            If the provider is unsupported.
        ImportError
            If the required SDK is not installed.
        """
        key = self._resolve_api_key()
        if not key:
            raise ValueError(
                f"No API key configured for provider={self._provider!r}"
            )

        if self._provider in ("openai", "custom"):
            return self._chat_openai(prompt, key)
        if self._provider == "anthropic":
            return self._chat_anthropic(prompt, key)

        raise ValueError(f"Unsupported provider: {self._provider!r}")

    def __repr__(self) -> str:
        return (
            f"RemoteLLMProvider(provider={self._provider!r}, "
            f"model={self._model!r})"
        )
