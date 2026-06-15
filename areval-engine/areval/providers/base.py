"""Provider abstract base classes.

Defines the core interfaces for pluggable model providers:
- ``LocalModelProvider``:  local LLM services (LM Studio, Ollama, etc.)
- ``RemoteLLMProvider``: remote LLM APIs (OpenAI, Anthropic, custom endpoints)

These ABCs enforce a consistent contract so that metrics / judges can
dispatch requests without coupling to a specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LocalModelProvider(ABC):
    """Abstract base for local model providers (LM Studio, Ollama, etc.).

    Tier-2 providers expose an OpenAI-compatible chat-completion endpoint
    and are probed via lightweight HTTP checks (no model loading).
    """

    name: str = "local_model"

    # ------------------------------------------------------------------
    #  Properties (must be overridden or set on instance)
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def timeout(self) -> float:
        """HTTP request timeout in seconds."""
        ...

    @property
    @abstractmethod
    def max_retries(self) -> int:
        """Number of retry attempts for transient failures."""
        ...

    # ------------------------------------------------------------------
    #  Lifecycle / capability checks
    # ------------------------------------------------------------------
    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the provider backend is reachable.

        MUST be a lightweight operation:
        - Only performs HTTP health / model-list probes.
        - NEVER loads a model into memory.
        - Result SHOULD be cached at module / instance level.

        Returns
        -------
        bool
            ``True`` if the service is responding.
        """
        ...

    @abstractmethod
    def chat_complete(self, prompt: str, **kwargs: str) -> str:
        """Send a prompt to the local LLM and return the raw text response.

        Parameters
        ----------
        prompt : str
            The full prompt to send.
        **kwargs : str
            Extra format kwargs forwarded to the prompt template
            (may be ignored by the provider but accepted for consistency).

        Returns
        -------
        str
            The model's raw text response.
        """
        ...


class RemoteLLMProvider(ABC):
    """Abstract base for remote LLM APIs (OpenAI, Anthropic, custom endpoints).

    Tier-3 providers handle complex reasoning tasks and are only used
    when API keys are configured.
    """

    name: str = "remote_llm"

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def timeout(self) -> float:
        """HTTP request timeout in seconds."""
        ...

    @property
    @abstractmethod
    def max_retries(self) -> int:
        """Number of retry attempts for transient failures."""
        ...

    # ------------------------------------------------------------------
    #  Lifecycle / capability checks
    # ------------------------------------------------------------------
    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if a valid API key is set AND the endpoint is reachable.

        MUST be a lightweight operation (HTTP probe, no model loading).
        Result SHOULD be cached.
        """
        ...

    @abstractmethod
    def chat_complete(self, prompt: str) -> str:
        """Send a prompt to the remote LLM and return the raw text response.

        Parameters
        ----------
        prompt : str
            The full prompt (system + user combined).

        Returns
        -------
        str
            The model's raw text response.
        """
        ...
