"""Provider protocol — the contract every LLM wrapper must satisfy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ProviderResponse:
    """Raw response from a provider."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def name(self) -> str: ...

    async def complete(self, prompt: str, model: str) -> ProviderResponse:
        """Send a prompt and return the response."""
        ...
