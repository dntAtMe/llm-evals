"""Async execution engine — runs prompts x models x N runs."""

from __future__ import annotations

import asyncio
import time

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from llm_eval.engine.retry import with_retry
from llm_eval.models import EvalRun, LLMResponse, Scenario
from llm_eval.providers import get_provider

console = Console(stderr=True)

# Per-provider semaphores to respect rate limits
_semaphores: dict[str, asyncio.Semaphore] = {}

PROVIDER_CONCURRENCY = 5


def _get_semaphore(provider_name: str) -> asyncio.Semaphore:
    if provider_name not in _semaphores:
        _semaphores[provider_name] = asyncio.Semaphore(PROVIDER_CONCURRENCY)
    return _semaphores[provider_name]


async def _execute_single(
    prompt_id: str,
    language: str,
    prompt_text: str,
    provider_name: str,
    model: str,
    run_index: int,
) -> LLMResponse:
    """Execute a single prompt against a single model."""
    provider = get_provider(provider_name)
    sem = _get_semaphore(provider_name)

    async with sem:
        label = f"{provider_name}/{model} {prompt_id}[{language}] run={run_index}"
        start = time.perf_counter()
        result = await with_retry(
            lambda: provider.complete(prompt_text, model),
            label=label,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

    return LLMResponse(
        prompt_id=prompt_id,
        language=language,
        provider=provider_name,
        model=model,
        run_index=run_index,
        prompt_text=prompt_text,
        response_text=result.text,
        latency_ms=elapsed_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


async def run_scenario(scenario: Scenario) -> EvalRun:
    """Run all prompts x models x runs concurrently and return an EvalRun."""
    tasks: list[asyncio.Task[LLMResponse]] = []

    # Build all tasks
    for prompt in scenario.prompts:
        for lang, text in prompt.variants.items():
            for model_spec in scenario.models:
                for run_idx in range(scenario.runs_per_prompt):
                    task = asyncio.create_task(
                        _execute_single(
                            prompt_id=prompt.id,
                            language=lang,
                            prompt_text=text,
                            provider_name=model_spec.provider,
                            model=model_spec.model,
                            run_index=run_idx,
                        )
                    )
                    tasks.append(task)

    total = len(tasks)
    console.print(f"\n[bold]Running {total} API calls...[/bold]\n")

    responses: list[LLMResponse] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        pbar = progress.add_task("Executing", total=total)
        for coro in asyncio.as_completed(tasks):
            resp = await coro
            responses.append(resp)
            progress.advance(pbar)

    # Sort for deterministic output
    responses.sort(key=lambda r: (r.prompt_id, r.provider, r.model, r.language, r.run_index))

    return EvalRun(
        scenario_name=scenario.name,
        responses=responses,
        metadata={
            "models": [f"{m.provider}/{m.model}" for m in scenario.models],
            "runs_per_prompt": scenario.runs_per_prompt,
            "total_calls": total,
        },
    )
