"""Anthropic Claude provider."""

from __future__ import annotations

import anthropic

from llm_eval.config import get_settings
from llm_eval.providers.base import ProviderResponse


class AnthropicProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(self, prompt: str, model: str) -> ProviderResponse:
        response = await self._client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return ProviderResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
