"""LLM-as-judge: quality scoring and term extraction with English translation."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from rich.console import Console

from llm_eval.engine.retry import with_retry
from llm_eval.models import (
    CrossLanguageComparison,
    CrossLanguageDiff,
    ExtractedTerms,
    JudgeConfig,
    JudgeResult,
    JudgeScore,
    LLMResponse,
    ScoreDimension,
    Scenario,
)
from llm_eval.providers import get_provider

console = Console(stderr=True)

_DEFAULT_DIMENSIONS: list[ScoreDimension] = [
    ScoreDimension(name="specificity", description="Does it give concrete, specific advice (varieties, dates, measurements)?"),
    ScoreDimension(name="actionability", description="Can someone follow this advice step-by-step?"),
    ScoreDimension(name="accuracy", description="Is the horticultural information correct?"),
    ScoreDimension(name="completeness", description="Does it cover the main aspects of the question?"),
]

TERM_EXTRACTION_PROMPT = """\
Extract key gardening terms from this LLM response. \
The response may be in any language — translate ALL extracted terms to English.

**Response:**
{response}

Extract terms into these categories:
- **plants**: specific plant species, varieties, or cultivars mentioned
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
    "plants": ["term1", "term2"],
    "techniques": ["term1", "term2"],
    "timing": ["term1", "term2"],
    "tools_products": ["term1", "term2"]
}}
"""


def _build_quality_prompt(dimensions: list[ScoreDimension]) -> str:
    """Build a quality-evaluation prompt template from scoring dimensions."""
    dims_lines = "\n".join(
        f"- **{d.name.replace('_', ' ').title()}**: {d.description}"
        for d in dimensions
    )
    json_keys = ", ".join(f'"{d.name}": N' for d in dimensions)
    return (
        "You are evaluating an LLM's response to a question.\n\n"
        "**Question ({language}):**\n{prompt}\n\n"
        "**Response:**\n{response}\n\n"
        f"Rate the response on these dimensions (1=poor, 5=excellent):\n{dims_lines}\n\n"
        "Respond with ONLY a JSON object:\n"
        "{{" + json_keys + "}}"
    )


def _resolve_prompts(judge_config: JudgeConfig | None) -> tuple[str, str]:
    """Return (quality_prompt_template, term_extraction_prompt_template) to use."""
    if judge_config and judge_config.quality_prompt:
        quality = judge_config.quality_prompt
    elif judge_config and judge_config.dimensions:
        quality = _build_quality_prompt(judge_config.dimensions)
    else:
        quality = _build_quality_prompt(_DEFAULT_DIMENSIONS)

    term_extraction = (
        judge_config.term_extraction_prompt
        if judge_config and judge_config.term_extraction_prompt
        else TERM_EXTRACTION_PROMPT
    )
    return quality, term_extraction


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
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from potentially messy LLM output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } in the text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


CROSS_LANGUAGE_PROMPT = """\
You are comparing LLM responses to the SAME gardening question asked in different languages.
Your goal is to find semantic differences — not translation artifacts, but actual differences \
in advice, assumptions, recommendations, or information.

**Question (asked identically in {n_langs} languages):**
{question_en}

{responses_block}

Analyze the responses and identify concrete semantic differences. For each difference, classify it:
- **recommendations**: different plants, products, or methods suggested
- **assumptions**: different climate, region, or growing conditions assumed
- **omissions**: information present in some languages but missing in others
- **emphasis**: same topic but different priority or depth given to it

Also provide a 2-3 sentence summary of the overall divergence.

Respond with ONLY a JSON object:
{{
    "differences": [
        {{
            "category": "recommendations|assumptions|omissions|emphasis",
            "description": "what differs (in English)",
            "languages_affected": ["lang1", "lang2"]
        }}
    ],
    "summary": "overall summary of divergence"
}}
"""


async def _compare_cross_language(
    judge_provider: str,
    judge_model: str,
    prompt_id: str,
    provider: str,
    model: str,
    lang_responses: dict[str, LLMResponse],
    judge_config: JudgeConfig | None = None,
) -> CrossLanguageComparison:
    """Compare responses across languages for the same prompt+model."""
    languages = sorted(lang_responses.keys())

    # Build the responses block
    parts = []
    for lang in languages:
        resp = lang_responses[lang]
        parts.append(f"**Response in {lang.upper()}:**\n{resp.response_text}")
    responses_block = "\n\n---\n\n".join(parts)

    # Use the first language's prompt text as reference
    first_resp = lang_responses[languages[0]]
    question_en = first_resp.prompt_text

    cross_prompt = (
        judge_config.cross_language_prompt
        if judge_config and judge_config.cross_language_prompt
        else CROSS_LANGUAGE_PROMPT
    )

    data = await _judge_single(
        judge_provider,
        judge_model,
        cross_prompt,
        n_langs=str(len(languages)),
        question_en=question_en,
        responses_block=responses_block,
    )

    diffs = [CrossLanguageDiff(**d) for d in data.get("differences", [])]

    return CrossLanguageComparison(
        prompt_id=prompt_id,
        provider=provider,
        model=model,
        languages=languages,
        differences=diffs,
        summary=data.get("summary", ""),
    )


async def _evaluate_group(
    judge_provider: str,
    judge_model: str,
    prompt_id: str,
    provider: str,
    model: str,
    language: str,
    responses: list[LLMResponse],
    judge_config: JudgeConfig | None = None,
) -> JudgeResult:
    """Evaluate a group of responses (same prompt/model/language)."""
    rep = responses[0]
    quality_prompt, term_prompt = _resolve_prompts(judge_config)

    quality_data, terms_data = await asyncio.gather(
        _judge_single(
            judge_provider,
            judge_model,
            quality_prompt,
            language=language,
            prompt=rep.prompt_text,
            response=rep.response_text,
        ),
        _judge_single(
            judge_provider,
            judge_model,
            term_prompt,
            response=rep.response_text,
        ),
    )

    return JudgeResult(
        prompt_id=prompt_id,
        provider=provider,
        model=model,
        language=language,
        quality_scores=JudgeScore(scores=quality_data),
        extracted_terms=ExtractedTerms(**terms_data),
    )


async def run_judge_analysis(
    responses: list[LLMResponse],
    scenario: Scenario,
) -> tuple[list[JudgeResult], list[CrossLanguageComparison]]:
    """Run LLM judge analysis: per-language + cross-language comparison."""
    if not scenario.judge:
        console.print("[yellow]No judge configured, skipping judge analysis.[/yellow]")
        return [], []

    judge_provider = scenario.judge.provider
    judge_model = scenario.judge.model

    # --- Phase 1: Per-language evaluation ---
    groups: dict[tuple[str, str, str, str], list[LLMResponse]] = defaultdict(list)
    for r in responses:
        groups[(r.prompt_id, r.provider, r.model, r.language)].append(r)

    console.print(
        f"\n[bold]Running judge analysis ({len(groups)} groups) "
        f"with {judge_provider}/{judge_model}...[/bold]\n"
    )

    judge_config = scenario.judge_config

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
                judge_config=judge_config,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    judge_results: list[JudgeResult] = []
    keys = list(groups.keys())
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            prompt_id, prov, mdl, lang = keys[i]
            console.print(
                f"[red]Judge error [{prompt_id}/{lang}]: {r}[/red]"
            )
        else:
            judge_results.append(r)

    # --- Phase 2: Cross-language comparison ---
    # Group by (prompt_id, provider, model) — pick one representative per language
    cross_groups: dict[tuple[str, str, str], dict[str, LLMResponse]] = defaultdict(dict)
    for r in responses:
        key = (r.prompt_id, r.provider, r.model)
        if r.language not in cross_groups[key]:
            cross_groups[key][r.language] = r

    # Only compare groups with 2+ languages
    cross_tasks = []
    cross_keys = []
    for (prompt_id, provider, model), lang_responses in cross_groups.items():
        if len(lang_responses) < 2:
            continue
        cross_keys.append((prompt_id, provider, model))
        cross_tasks.append(
            _compare_cross_language(
                judge_provider, judge_model,
                prompt_id, provider, model, lang_responses,
                judge_config=judge_config,
            )
        )

    comparisons: list[CrossLanguageComparison] = []
    if cross_tasks:
        console.print(
            f"[bold]Running cross-language comparison "
            f"({len(cross_tasks)} groups)...[/bold]\n"
        )
        cross_results = await asyncio.gather(*cross_tasks, return_exceptions=True)
        for i, r in enumerate(cross_results):
            if isinstance(r, Exception):
                pid, prov, mdl = cross_keys[i]
                console.print(
                    f"[red]Cross-language error [{pid}/{prov}]: {r}[/red]"
                )
            else:
                comparisons.append(r)

    return judge_results, comparisons
