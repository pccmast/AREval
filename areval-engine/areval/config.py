"""AREval provider configuration – auto-detect, profiles, and fallback.

Implements ADR-5: three-layer config (Env Vars > Config File > Auto-Detect)
with one-click profile switching via ``AREVAL_PROFILE``.

Usage::

    from areval.config import load_config, print_config

    cfg = load_config()
    print_config(cfg)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LMSTUDIO_URL = "http://localhost:12345/v1"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_REMOTE_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_MODEL = "qwen3-1.7b"

# Profiles → default local LLM URL
PROFILE_URL_MAP: dict[str, Optional[str]] = {
    "lmstudio": DEFAULT_LMSTUDIO_URL,
    "ollama": DEFAULT_OLLAMA_URL,
    "docker": "http://local-models:11434",
    "remote": None,
    "mock": None,
}

# Local endpoint candidates in priority order (auto-detect)
LOCAL_CANDIDATES: list[tuple[str, str]] = [
    ("lmstudio", f"{DEFAULT_LMSTUDIO_URL}/models"),
    ("ollama", f"{DEFAULT_OLLAMA_URL}/api/tags"),
    ("docker", "http://local-models:11434/api/tags"),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ProviderConfig:
    """Resolved provider configuration.

    Built by merging Env Vars → Config File → Auto-Detect (highest priority wins).
    """

    profile: str = "auto"
    local_llm_url: Optional[str] = None
    local_llm_model: Optional[str] = None
    remote_llm_url: Optional[str] = None
    remote_llm_model: str = DEFAULT_REMOTE_MODEL
    tier2_enabled: bool = True
    tier3_enabled: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _probe_http(url: str, timeout: float = 0.5) -> bool:
    """Lightweight HTTP reachability probe.  Any error → False."""
    try:
        import httpx

        r = httpx.get(url, timeout=timeout, follow_redirects=False)
        return r.status_code < 500
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError,
            httpx.RemoteProtocolError):
        return False
    except Exception:
        return False


def _auto_detect_profile() -> tuple[Optional[str], Optional[str]]:
    """Scan known local endpoints; return (profile_name, url) or (None, None)."""
    for name, url in LOCAL_CANDIDATES:
        if _probe_http(url, timeout=0.5):
            return name, url.rstrip("/models").rstrip("/api/tags")
    return None, None


def _load_yaml_config(path: str = "") -> dict[str, object]:
    """Load optional ``~/.areval/config.yaml``.  Returns empty dict on failure."""
    if not path:
        path = os.path.expanduser("~/.areval/config.yaml")
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def log_degradation(from_tier: str, to_tier: str, reason: str) -> None:
    """Unified degradation log (ADR-5).  Outputs to stderr."""
    print(
        f"[AREval] Tier {from_tier} → Tier {to_tier} | {reason}",
        file=sys.stderr,
    )


def load_config() -> ProviderConfig:
    """Merge Env Vars → Config File → Auto-Detect → return ``ProviderConfig``.

    Priority (highest first):
    1. Environment variables (``AREVAL_PROFILE``, ``AREVAL_LOCAL_LLM_URL``, etc.)
    2. ``~/.areval/config.yaml``
    3. Auto-detection (scan known local endpoints)
    """
    cfg = ProviderConfig()

    # --- Layer 1: Auto-Detect ---
    detected_profile, detected_url = _auto_detect_profile()
    remote_available = bool(os.environ.get("OPENAI_API_KEY"))

    if detected_profile:
        cfg.profile = detected_profile
        cfg.local_llm_url = detected_url
    elif remote_available:
        cfg.profile = "remote"
    else:
        cfg.profile = "mock"

    # --- Layer 2: Config File ---
    yaml_data = _load_yaml_config()
    if yaml_data:
        if "profile" in yaml_data:
            cfg.profile = str(yaml_data["profile"])
        providers = yaml_data.get("providers", {})
        if isinstance(providers, dict):
            llm = providers.get("local_llm", {})
            if isinstance(llm, dict):
                if llm.get("url"):
                    cfg.local_llm_url = str(llm["url"])
                if llm.get("model"):
                    cfg.local_llm_model = str(llm["model"])
            remote = providers.get("remote_llm", {})
            if isinstance(remote, dict):
                if remote.get("url"):
                    cfg.remote_llm_url = str(remote["url"])
                if remote.get("model"):
                    cfg.remote_llm_model = str(remote["model"])

    # --- Layer 3: Environment Variables ---
    env_profile = os.environ.get("AREVAL_PROFILE")
    if env_profile:
        cfg.profile = env_profile

    env_local_url = os.environ.get("AREVAL_LOCAL_LLM_URL")
    if env_local_url:
        cfg.local_llm_url = env_local_url

    env_local_model = os.environ.get("AREVAL_LOCAL_LLM_MODEL")
    if env_local_model:
        cfg.local_llm_model = env_local_model

    env_remote_url = os.environ.get("AREVAL_LLM_BASE_URL")
    if env_remote_url:
        cfg.remote_llm_url = env_remote_url

    # --- Post-processing: apply profile defaults ---
    if cfg.local_llm_url is None and cfg.profile in PROFILE_URL_MAP:
        cfg.local_llm_url = PROFILE_URL_MAP[cfg.profile]

    if cfg.local_llm_model is None:
        cfg.local_llm_model = DEFAULT_LOCAL_MODEL

    if cfg.profile in ("mock",):
        cfg.tier2_enabled = False
        cfg.tier3_enabled = False
    elif cfg.profile in ("remote",):
        cfg.tier2_enabled = False

    return cfg


def print_config(config: ProviderConfig) -> None:
    """Print a human-readable configuration summary on startup."""
    lines = [
        f"[AREval] Profile: {config.profile}",
        "[AREval] Tier 1 (pure code): always available",
    ]
    if config.tier2_enabled:
        url = config.local_llm_url or "auto-detect"
        model = config.local_llm_model or DEFAULT_LOCAL_MODEL
        lines.append(
            f"[AREval] Tier 2 (local LLM):  {model} @ {url}"
        )
    else:
        lines.append("[AREval] Tier 2 (local LLM):  disabled")

    if config.tier3_enabled:
        url = config.remote_llm_url or DEFAULT_OPENAI_URL
        model = config.remote_llm_model
        lines.append(
            f"[AREval] Tier 3 (remote LLM): {model} @ {url}"
        )
    else:
        lines.append("[AREval] Tier 3 (remote LLM): disabled")

    for line in lines:
        print(line)
