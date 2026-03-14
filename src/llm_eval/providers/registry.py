"""Provider factory and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_eval.providers.base import LLMProvider

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register a provider class."""

    def wrapper(cls: type):
        _REGISTRY[name] = cls
        return cls

    return wrapper


def _ensure_builtins() -> None:
    """Lazily register built-in providers."""
    if "anthropic" not in _REGISTRY:
        from llm_eval.providers.anthropic import AnthropicProvider

        _REGISTRY["anthropic"] = AnthropicProvider
    if "openai" not in _REGISTRY:
        from llm_eval.providers.openai import OpenAIProvider

        _REGISTRY["openai"] = OpenAIProvider
    if "gemini" not in _REGISTRY:
        from llm_eval.providers.gemini import GeminiProvider

        _REGISTRY["gemini"] = GeminiProvider


_instances: dict[str, LLMProvider] = {}


def get_provider(name: str) -> LLMProvider:
    """Get or create a singleton provider instance."""
    _ensure_builtins()
    if name not in _instances:
        if name not in _REGISTRY:
            raise ValueError(f"Unknown provider: {name}. Available: {list(_REGISTRY.keys())}")
        _instances[name] = _REGISTRY[name]()
    return _instances[name]


def list_providers() -> list[str]:
    _ensure_builtins()
    return list(_REGISTRY.keys())
