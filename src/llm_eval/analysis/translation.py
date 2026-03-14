"""Optional translation bridge for comparison — uses LLM for translation."""

from __future__ import annotations

import asyncio

from llm_eval.engine.retry import with_retry
from llm_eval.providers import get_provider

TRANSLATE_PROMPT = """\
Translate the following text from {source_lang} to {target_lang}. \
Provide only the translation, no explanations.

Text:
{text}
"""


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    provider_name: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Translate text using an LLM provider."""
    provider = get_provider(provider_name)
    prompt = TRANSLATE_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
        text=text,
    )
    result = await with_retry(
        lambda: provider.complete(prompt, model),
        label=f"translate:{source_lang}->{target_lang}",
    )
    return result.text.strip()


async def translate_batch(
    texts: list[str],
    source_lang: str,
    target_lang: str,
    provider_name: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
) -> list[str]:
    """Translate a batch of texts concurrently."""
    tasks = [
        translate_text(t, source_lang, target_lang, provider_name, model)
        for t in texts
    ]
    return await asyncio.gather(*tasks)
