"""AREval provider layer – pluggable model backends.

This module supplies Tier-2 (local) and Tier-3 (remote) model providers
with a consistent interface so that metrics and judges are backend-agnostic.

Concrete provider classes are exported for convenience; abstract base classes
are still accessible via ``areval.providers.base``.
"""

from areval.providers.base import LocalModelProvider as _LocalModelProviderABC
from areval.providers.local_llm import LocalLLMProvider
from areval.providers.remote_llm import RemoteLLMProvider

__all__ = [
    "_LocalModelProviderABC",
    "LocalLLMProvider",
    "RemoteLLMProvider",
]
