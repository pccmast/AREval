"""Tests for areval.providers module.

Covers:
- Provider instantiation and property defaults
- ``is_available()`` lightweight behaviour (no model loading)
- Delegation contracts (LocalLLMProvider / RemoteLLMProvider)
- Configuration helpers (load_config, log_degradation, print_config)
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from areval.config import (
    ProviderConfig,
    load_config,
    log_degradation,
    print_config,
    _auto_detect_profile,
)
from areval.providers.base import LocalModelProvider, RemoteLLMProvider
from areval.providers.local_llm import LocalLLMProvider as LocalImpl
from areval.providers.remote_llm import RemoteLLMProvider as RemoteImpl


# ============================================================================
#  Test: base ABCs enforce contracts
# ============================================================================
class TestBaseABC:
    def test_local_model_provider_is_abstract(self):
        """LocalModelProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LocalModelProvider()  # type: ignore[abstract]

    def test_remote_llm_provider_is_abstract(self):
        """RemoteLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RemoteLLMProvider()  # type: ignore[abstract]

    def test_local_impl_is_instance(self):
        """LocalLLMProvider is a valid concrete subclass."""
        p = LocalImpl()
        assert isinstance(p, LocalModelProvider)

    def test_remote_impl_is_instance(self):
        """RemoteLLMProvider is a valid concrete subclass."""
        p = RemoteImpl(provider="openai")
        assert isinstance(p, RemoteLLMProvider)


# ============================================================================
#  Test: LocalLLMProvider
# ============================================================================
class TestLocalLLMProvider:
    def test_defaults(self):
        p = LocalImpl()
        assert p.model == "qwen3-1.7b"
        assert p.base_url == "http://localhost:12345/v1"
        assert p.timeout == 30.0
        assert p.max_retries == 2

    def test_custom_params(self):
        p = LocalImpl(
            base_url="http://gpu:8000/v1",
            model="custom-model",
            timeout=10.0,
            max_retries=1,
            api_key="secret",
        )
        assert p.base_url == "http://gpu:8000/v1"
        assert p.model == "custom-model"
        assert p.timeout == 10.0
        assert p.max_retries == 1

    def test_base_url_strips_trailing_slash(self):
        p = LocalImpl(base_url="http://localhost:12345/v1/")
        assert p.base_url == "http://localhost:12345/v1"

    def test_is_available_cached(self):
        """is_available should cache result per URL."""
        # Force cache reset
        import areval.providers.local_llm as mod

        mod._availability_cache = None
        mod._cached_base_url = None

        with patch.object(mod, "_probe_http", return_value=True) as mock_probe:
            p = LocalImpl()
            assert p.is_available() is True
            assert p.is_available() is True
            # Probe should only be called once due to caching
            assert mock_probe.call_count == 1

    def test_is_available_reprobes_on_url_change(self):
        """Changing base_url should re-probe."""
        import areval.providers.local_llm as mod

        mod._availability_cache = None
        mod._cached_base_url = None

        with patch.object(mod, "_probe_http", return_value=True) as mock_probe:
            p1 = LocalImpl(base_url="http://localhost:12345/v1")
            assert p1.is_available() is True
            p2 = LocalImpl(base_url="http://other:8000/v1")
            assert p2.is_available() is True
            assert mock_probe.call_count == 2

    def test_is_available_returns_false_when_unreachable(self):
        import areval.providers.local_llm as mod

        mod._availability_cache = None
        mod._cached_base_url = None

        with patch.object(mod, "_probe_http", return_value=False):
            p = LocalImpl()
            assert p.is_available() is False

    def test_chat_complete_calls_openai_sdk(self):
        """chat_complete delegates to the OpenAI client correctly."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello, world!"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        # chat_complete uses `from openai import OpenAI` internally,
        # so we must patch at the source package level.
        with patch(
            "openai.OpenAI", return_value=mock_client
        ) as mock_openai_cls:
            p = LocalImpl()
            result = p.chat_complete("Hi")

        assert result == "Hello, world!"
        mock_openai_cls.assert_called_once()
        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:12345/v1"
        assert call_kwargs["timeout"] == 30.0
        assert call_kwargs["max_retries"] == 2

    def test_chat_complete_none_content(self):
        """Handle None content gracefully."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        with patch("openai.OpenAI", return_value=mock_client):
            p = LocalImpl()
            result = p.chat_complete("Hi")

        assert result == ""

    def test_repr(self):
        p = LocalImpl(base_url="http://gpu:8000/v1", model="qwen3-1.7b")
        r = repr(p)
        assert "http://gpu:8000/v1" in r
        assert "qwen3-1.7b" in r


# ============================================================================
#  Test: RemoteLLMProvider
# ============================================================================
class TestRemoteLLMProvider:
    def test_defaults(self):
        p = RemoteImpl(provider="openai", api_key="sk-test")
        assert p.model == "gpt-4o-mini"
        assert p.timeout == 60.0
        assert p.max_retries == 3
        assert p.provider_name == "openai"

    def test_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            RemoteImpl(provider="unknown")

    def test_custom_base_url(self):
        p = RemoteImpl(
            provider="custom",
            api_key="sk-test",
            base_url="https://my-llm.example.com/v1",
        )
        assert p._base_url == "https://my-llm.example.com/v1"

    def test_is_available_false_without_key(self, monkeypatch):
        """Without API key, is_available returns False."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = RemoteImpl(provider="openai")
        assert p.is_available() is False

    def test_is_available_with_key_but_unreachable(self, monkeypatch):
        """With key but unreachable endpoint → False."""
        p = RemoteImpl(provider="openai", api_key="sk-test")
        with patch.object(RemoteImpl, "_probe_http", return_value=False):
            assert p.is_available() is False

    def test_is_available_with_key_and_reachable(self):
        """With key + reachable endpoint → True."""
        p = RemoteImpl(provider="openai", api_key="sk-test")
        with patch.object(RemoteImpl, "_probe_http", return_value=True):
            assert p.is_available() is True

    def test_anthropic_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        p = RemoteImpl(provider="anthropic")
        assert p._resolve_api_key() == "sk-ant-test"

    def test_openai_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        p = RemoteImpl(provider="openai")
        assert p._resolve_api_key() == "sk-openai-test"

    def test_chat_complete_openai(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "evaluation result"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        # _chat_openai does `from openai import OpenAI` internally
        with patch(
            "openai.OpenAI", return_value=mock_client
        ):
            p = RemoteImpl(provider="openai", api_key="sk-test")
            result = p.chat_complete("evaluate this")

        assert result == "evaluation result"

    def test_chat_complete_anthropic(self):
        mock_client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "anthropic result"
        mock_client.messages.create.return_value.content = [mock_content]

        # Patch sys.modules so that `from anthropic import Anthropic` succeeds
        # even when the anthropic package is not installed.
        fake_anthropic = MagicMock()
        fake_anthropic.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            p = RemoteImpl(provider="anthropic", api_key="sk-test")
            result = p.chat_complete("evaluate this")

        assert result == "anthropic result"

    def test_chat_complete_without_key_raises(self):
        p = RemoteImpl(provider="openai")
        with pytest.raises(ValueError, match="No API key"):
            p.chat_complete("test")

    def test_repr(self):
        p = RemoteImpl(provider="openai", api_key="sk-test", model="gpt-4")
        r = repr(p)
        assert "openai" in r
        assert "gpt-4" in r


# ============================================================================
#  Test: config.py helpers
# ============================================================================
class TestConfig:
    def test_load_config_default_is_mock_when_nothing_available(self, monkeypatch):
        """Without any API keys or local endpoints → profile = mock."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AREVAL_PROFILE", raising=False)
        monkeypatch.delenv("AREVAL_LOCAL_LLM_URL", raising=False)

        with patch("areval.config._auto_detect_profile", return_value=(None, None)):
            with patch("areval.config._load_yaml_config", return_value={}):
                cfg = load_config()
                assert cfg.profile == "mock"
                assert cfg.tier2_enabled is False
                assert cfg.tier3_enabled is False

    def test_load_config_with_env_profile(self):
        """AREVAL_PROFILE env var takes highest priority."""
        with patch.dict(os.environ, {"AREVAL_PROFILE": "lmstudio"}, clear=False):
            with patch("areval.config._load_yaml_config", return_value={}):
                cfg = load_config()
                assert cfg.profile == "lmstudio"
                assert cfg.local_llm_url == "http://localhost:12345/v1"
                assert cfg.tier2_enabled is True

    def test_load_config_env_overrides_url(self):
        """AREVAL_LOCAL_LLM_URL overrides auto-detect."""
        with patch.dict(
            os.environ,
            {
                "AREVAL_PROFILE": "lmstudio",
                "AREVAL_LOCAL_LLM_URL": "http://custom:9999/v1",
            },
            clear=False,
        ):
            with patch("areval.config._load_yaml_config", return_value={}):
                cfg = load_config()
                assert cfg.local_llm_url == "http://custom:9999/v1"

    def test_log_degradation(self, capsys):
        """log_degradation writes to stderr."""
        log_degradation("2", "mock", "connection refused")
        captured = capsys.readouterr()
        assert "Tier 2 → Tier mock" in captured.err
        assert "connection refused" in captured.err

    def test_print_config_smoke(self, capsys):
        """print_config should not raise."""
        cfg = ProviderConfig(
            profile="lmstudio",
            local_llm_url="http://localhost:12345/v1",
            local_llm_model="qwen3-1.7b",
            tier2_enabled=True,
            tier3_enabled=False,
        )
        print_config(cfg)
        captured = capsys.readouterr()
        assert "lmstudio" in captured.out
        assert "qwen3-1.7b" in captured.out
        assert "Tier 3" in captured.out

    def test_auto_detect_profile_returns_none_when_nothing(self):
        """No local endpoints → (None, None)."""
        with patch("areval.config._probe_http", return_value=False):
            profile, url = _auto_detect_profile()
            assert profile is None
            assert url is None

    def test_auto_detect_profile_finds_lmstudio(self):
        """First candidate (lmstudio) wins."""
        with patch("areval.config._probe_http", return_value=True):
            profile, url = _auto_detect_profile()
            assert profile == "lmstudio"
            assert url == "http://localhost:12345/v1"


# ============================================================================
#  conftest hooks for provider tests that require real backends
# ============================================================================
@pytest.fixture
def reset_local_cache():
    """Reset local_llm availability cache before each test that needs it."""
    import areval.providers.local_llm as mod

    mod._availability_cache = None
    mod._cached_base_url = None
    yield
    mod._availability_cache = None
    mod._cached_base_url = None
