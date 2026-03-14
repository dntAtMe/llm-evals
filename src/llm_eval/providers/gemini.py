"""Google Gemini provider (via OpenAI-compatible API)."""

from __future__ import annotations

import openai

from llm_eval.config import get_settings
from llm_eval.providers.base import ProviderResponse


class GeminiProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = openai.AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    @property
    def name(self) -> str:
        return "gemini"

    async def complete(self, prompt: str, model: str) -> ProviderResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        choice = response.choices[0]
        usage = response.usage
        return ProviderResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
