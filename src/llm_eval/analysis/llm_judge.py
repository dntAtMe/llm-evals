"""LLM-as-judge: quality scoring and term extraction with English translation."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from rich.console import Console

from llm_eval.engine.retry import with_retry
from llm_eval.models import (
    ExtractedTerms,
    JudgeResult,
    JudgeScore,
    LLMResponse,
    Scenario,
)
from llm_eval.providers import get_provider

console = Console(stderr=True)

QUALITY_PROMPT = """\
You are evaluating an LLM's response to a gardening question.

**Question ({language}):**
{prompt}

**Response:**
{response}

Rate the response on these dimensions (1=poor, 5=excellent):
- **Specificity**: Does it give concrete, specific advice (varieties, dates, measurements)?
- **Actionability**: Can someone follow this advice step-by-step?
- **Accuracy**: Is the horticultural information correct?
- **Completeness**: Does it cover the main aspects of the question?

Respond with ONLY a JSON object:
{{"specificity": N, "actionability": N, "accuracy": N, "completeness": N}}
"""

TERM_EXTRACTION_PROMPT = """\
Extract key gardening terms from this LLM response. \
The response may be in any language — translate ALL extracted terms to English.

**Response:**
{response}

Extract terms into these categories:
- **varieties**: specific plant varieties, cultivars, or species mentioned
- **techniques**: gardening methods, practices, or approaches
- **timing**: planting dates, seasons, harvest periods, calendar references
- **tools_products**: tools, fertilizers, soil amendments, products

Rules:
- Translate everything to English (e.g. "Zimni Ogrodnicy" → "Ice Saints")
- Use lowercase
- Keep variety names recognizable (e.g. "San Marzano", "Roma")
- Be specific: "add compost" → technique "composting", not just "soil amendment"

Respond with ONLY a JSON object:
{{
    "varieties": ["term1", "term2"],
    "techniques": ["term1", "term2"],
    "timing": ["term1", "term2"],
    "tools_products": ["term1", "term2"]
}}
"""


async def _judge_single(
    provider_name: str,
    model: str,
    prompt_template: str,
    **kwargs: str,
) -> dict:
    """Call the judge model with a structured prompt and parse JSON response."""
    provider = get_provider(provider_name)
    prompt = prompt_template.format(**kwargs)

    result = await with_retry(
        lambda: provider.complete(prompt, model),
        label=f"judge:{provider_name}/{model}",
        max_retries=2,
    )

    text = result.text.strip()
    # Extract JSON from response (handle markdown code blocks)
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    return json.loads(text)


async def _evaluate_group(
    judge_provider: str,
    judge_model: str,
    prompt_id: str,
    provider: str,
    model: str,
    language: str,
    responses: list[LLMResponse],
) -> JudgeResult:
    """Evaluate a group of responses (same prompt/model/language)."""
    rep = responses[0]

    quality_data, terms_data = await asyncio.gather(
        _judge_single(
            judge_provider,
            judge_model,
            QUALITY_PROMPT,
            language=language,
            prompt=rep.prompt_text,
            response=rep.response_text,
        ),
        _judge_single(
            judge_provider,
            judge_model,
            TERM_EXTRACTION_PROMPT,
            response=rep.response_text,
        ),
    )

    return JudgeResult(
        prompt_id=prompt_id,
        provider=provider,
        model=model,
        language=language,
        quality_scores=JudgeScore(**quality_data),
        extracted_terms=ExtractedTerms(**terms_data),
    )


async def run_judge_analysis(
    responses: list[LLMResponse],
    scenario: Scenario,
) -> list[JudgeResult]:
    """Run LLM judge analysis on all response groups."""
    if not scenario.judge:
        console.print("[yellow]No judge configured, skipping judge analysis.[/yellow]")
        return []

    judge_provider = scenario.judge.provider
    judge_model = scenario.judge.model

    # Group by (prompt_id, provider, model, language)
    groups: dict[tuple[str, str, str, str], list[LLMResponse]] = defaultdict(list)
    for r in responses:
        groups[(r.prompt_id, r.provider, r.model, r.language)].append(r)

    console.print(
        f"\n[bold]Running judge analysis ({len(groups)} groups) "
        f"with {judge_provider}/{judge_model}...[/bold]\n"
    )

    tasks = []
    for (prompt_id, provider, model, language), group_responses in groups.items():
        tasks.append(
            _evaluate_group(
                judge_provider,
                judge_model,
                prompt_id,
                provider,
                model,
                language,
                group_responses,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    judge_results: list[JudgeResult] = []
    for r in results:
        if isinstance(r, Exception):
            console.print(f"[red]Judge error: {r}[/red]")
        else:
            judge_results.append(r)

    return judge_results
