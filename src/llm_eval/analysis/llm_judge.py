"""LLM-as-judge: quality scoring and term extraction with English translation."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Literal

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
    Scenario,
    ScoreDimension,
)
from llm_eval.providers import get_provider

console = Console(stderr=True)

_DEFAULT_DIMENSIONS: list[ScoreDimension] = [
    ScoreDimension(
        name="specificity",
        description=(
            "Does it give concrete, specific advice (varieties, dates, measurements)?"
        ),
    ),
    ScoreDimension(
        name="actionability",
        description="Can someone follow this advice step-by-step?",
    ),
    ScoreDimension(
        name="accuracy",
        description="Is the horticultural information correct?",
    ),
    ScoreDimension(
        name="completeness",
        description="Does it cover the main aspects of the question?",
    ),
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

# Prepended when using a shared base template (cross_language_prompt or default) for concat mode.
_CROSS_LANG_CONCAT_PREAMBLE = (
    "Context: Below are MULTIPLE independent model outputs per language, "
    "labeled Run 1, Run 2, etc. Compare across languages and account for "
    "within-language variance when judging differences.\n\n"
)

# Prepended for summarized mode when using a shared base template.
_CROSS_LANG_SUMMARY_PREAMBLE = (
    "Context: Below each language is an English SUMMARY of multiple model runs in that language "
    "(not the raw outputs). Compare across languages based on these summaries.\n\n"
)


def _base_cross_language_template(judge_config: JudgeConfig | None) -> str:
    if judge_config and judge_config.cross_language_prompt:
        return judge_config.cross_language_prompt
    return CROSS_LANGUAGE_PROMPT


def _resolve_concat_cross_prompt(judge_config: JudgeConfig | None) -> str:
    if judge_config and judge_config.cross_language_concat_prompt:
        return judge_config.cross_language_concat_prompt
    return _CROSS_LANG_CONCAT_PREAMBLE + _base_cross_language_template(judge_config)


def _resolve_summarized_cross_prompt(judge_config: JudgeConfig | None) -> str:
    if judge_config and judge_config.cross_language_summarized_prompt:
        return judge_config.cross_language_summarized_prompt
    return _CROSS_LANG_SUMMARY_PREAMBLE + _base_cross_language_template(judge_config)


SUMMARIZE_RUNS_PROMPT = """\
You are synthesizing {n_runs} independent LLM responses to the same question
(language code: {language}).
The responses may differ due to randomness or substantive disagreement.

**Question:**
{question}

**Responses (labeled):**
{responses_block}

Write a unified summary in English that:
- Captures stable, recurring advice across runs
- Briefly notes important disagreements between runs, if any
- Ignores trivial wording differences

Write 2–6 short paragraphs of prose. Do not use JSON or bullet lists as the primary structure.
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
    return _extract_json(text)


async def _complete_plain(
    provider_name: str,
    model: str,
    prompt: str,
) -> str:
    """Call the judge model and return raw text (no JSON parse)."""
    provider = get_provider(provider_name)
    result = await with_retry(
        lambda: provider.complete(prompt, model),
        label=f"judge:{provider_name}/{model}",
        max_retries=2,
    )
    return result.text.strip()


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


def _build_concat_responses_block(by_lang: dict[str, list[LLMResponse]]) -> str:
    """One section per language: labeled runs, joined across languages."""
    languages = sorted(by_lang.keys())
    outer: list[str] = []
    for lang in languages:
        runs = sorted(by_lang[lang], key=lambda r: r.run_index)
        inner = "\n\n".join(
            f"**Run {i + 1}:**\n{r.response_text}" for i, r in enumerate(runs)
        )
        outer.append(f"**Responses in {lang.upper()} ({len(runs)} run(s)):**\n{inner}")
    return "\n\n---\n\n".join(outer)


def _build_summary_responses_block(lang_to_summary: dict[str, str]) -> str:
    """One section per language: English summary text."""
    languages = sorted(lang_to_summary.keys())
    parts = [f"**Summary for {lang.upper()}:**\n{lang_to_summary[lang]}" for lang in languages]
    return "\n\n---\n\n".join(parts)


async def _summarize_language_runs(
    judge_provider: str,
    judge_model: str,
    language: str,
    responses: list[LLMResponse],
    judge_config: JudgeConfig | None,
) -> str:
    """Synthesize multiple runs in one language into English prose; single run returns raw text."""
    runs = sorted(responses, key=lambda r: r.run_index)
    if len(runs) == 1:
        return runs[0].response_text

    tmpl = (
        judge_config.summarize_runs_prompt
        if judge_config and judge_config.summarize_runs_prompt
        else SUMMARIZE_RUNS_PROMPT
    )
    question = runs[0].prompt_text
    block = "\n\n".join(
        f"**Run {i + 1}:**\n{r.response_text}" for i, r in enumerate(runs)
    )
    prompt = tmpl.format(
        language=language,
        n_runs=str(len(runs)),
        question=question,
        responses_block=block,
    )
    return await _complete_plain(judge_provider, judge_model, prompt)


async def _compare_cross_language(
    judge_provider: str,
    judge_model: str,
    prompt_id: str,
    provider: str,
    model: str,
    languages: list[str],
    question_en: str,
    responses_block: str,
    cross_prompt_template: str,
    aggregation_mode: Literal["concatenated", "summarized"],
) -> CrossLanguageComparison:
    """Run cross-language JSON comparison with a pre-built responses block."""
    data = await _judge_single(
        judge_provider,
        judge_model,
        cross_prompt_template,
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
        aggregation_mode=aggregation_mode,
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


def _cross_language_groups(
    responses: list[LLMResponse],
) -> dict[tuple[str, str, str], dict[str, list[LLMResponse]]]:
    """(prompt_id, provider, model) -> language -> sorted list of runs."""
    cross_groups: dict[tuple[str, str, str], dict[str, list[LLMResponse]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in responses:
        key = (r.prompt_id, r.provider, r.model)
        cross_groups[key][r.language].append(r)
    for by_lang in cross_groups.values():
        for lang in by_lang:
            by_lang[lang].sort(key=lambda x: x.run_index)
    return cross_groups


async def run_judge_analysis(
    responses: list[LLMResponse],
    scenario: Scenario,
) -> tuple[list[JudgeResult], list[CrossLanguageComparison]]:
    """Run LLM judge analysis: per-language + dual cross-language (concat + summarized)."""
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

    # --- Phase 2: Cross-language (concatenated runs + summarized) ---
    cross_groups = _cross_language_groups(responses)
    group_items = [(k, v) for k, v in cross_groups.items() if len(v) >= 2]
    group_items.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))

    if not group_items:
        return judge_results, []

    concat_tpl = _resolve_concat_cross_prompt(judge_config)
    summarized_tpl = _resolve_summarized_cross_prompt(judge_config)

    # Per-(group, lang) summaries — parallel; concat comparisons can run alongside
    summarize_tasks: list[asyncio.Task[str]] = []
    summarize_meta: list[tuple[tuple[str, str, str], str]] = []
    for key, by_lang in group_items:
        for lang, runs in by_lang.items():
            summarize_meta.append((key, lang))
            summarize_tasks.append(
                asyncio.create_task(
                    _summarize_language_runs(
                        judge_provider,
                        judge_model,
                        lang,
                        runs,
                        judge_config,
                    )
                )
            )

    concat_tasks = []
    concat_meta: list[tuple[str, str, str]] = []
    for (prompt_id, provider, model), by_lang in group_items:
        concat_meta.append((prompt_id, provider, model))
        languages = sorted(by_lang.keys())
        question_en = next(iter(by_lang.values()))[0].prompt_text
        block = _build_concat_responses_block(by_lang)
        concat_tasks.append(
            _compare_cross_language(
                judge_provider,
                judge_model,
                prompt_id,
                provider,
                model,
                languages,
                question_en,
                block,
                concat_tpl,
                "concatenated",
            )
        )

    console.print(
        f"[bold]Running cross-language comparison: "
        f"{len(concat_tasks)} concat + {len(concat_tasks)} summarized "
        f"({len(summarize_tasks)} per-language summaries)...[/bold]\n"
    )

    concat_results, summ_text_results = await asyncio.gather(
        asyncio.gather(*concat_tasks, return_exceptions=True),
        asyncio.gather(*summarize_tasks, return_exceptions=True),
    )

    # Map (key, lang) -> summary text
    lang_summaries: dict[tuple[tuple[str, str, str], str], str] = {}
    for i, res in enumerate(summ_text_results):
        key, lang = summarize_meta[i]
        if isinstance(res, Exception):
            pid, prov, mdl = key
            console.print(
                f"[red]Summarize error [{pid}/{lang}/{prov}]: {res}[/red]"
            )
            lang_summaries[(key, lang)] = ""
        else:
            lang_summaries[(key, lang)] = res

    summary_compare_tasks = []
    for (prompt_id, provider, model), by_lang in group_items:
        key = (prompt_id, provider, model)
        languages = sorted(by_lang.keys())
        if any(not lang_summaries.get((key, L), "").strip() for L in languages):
            # Skip if any summary failed (empty)
            summary_compare_tasks.append(None)
            continue
        question_en = by_lang[languages[0]][0].prompt_text
        block = _build_summary_responses_block(
            {L: lang_summaries[(key, L)] for L in languages}
        )
        summary_compare_tasks.append(
            _compare_cross_language(
                judge_provider,
                judge_model,
                prompt_id,
                provider,
                model,
                languages,
                question_en,
                block,
                summarized_tpl,
                "summarized",
            )
        )

    summary_compare_results_raw = await asyncio.gather(
        *[t for t in summary_compare_tasks if t is not None],
        return_exceptions=True,
    )

    # Re-align summary results with group order (including skipped)
    summary_iter = iter(summary_compare_results_raw)
    summary_compare_results: list[CrossLanguageComparison | Exception | None] = []
    for t in summary_compare_tasks:
        if t is None:
            summary_compare_results.append(None)
        else:
            summary_compare_results.append(next(summary_iter))

    comparisons: list[CrossLanguageComparison] = []

    for i, (prompt_id, provider, model) in enumerate(concat_meta):
        cr = concat_results[i]
        if isinstance(cr, Exception):
            console.print(
                f"[red]Cross-language (concat) error [{prompt_id}/{provider}]: {cr}[/red]"
            )
        else:
            comparisons.append(cr)

        sr = summary_compare_results[i]
        if sr is None:
            console.print(
                f"[yellow]Cross-language (summarized) skipped [{prompt_id}/{provider}]: "
                "missing summary[/yellow]"
            )
        elif isinstance(sr, Exception):
            console.print(
                f"[red]Cross-language (summarized) error [{prompt_id}/{provider}]: {sr}[/red]"
            )
        else:
            comparisons.append(sr)

    return judge_results, comparisons
