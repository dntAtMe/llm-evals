"""Typer CLI — run, analyze, list-providers."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from llm_eval.config import get_settings, load_scenario
from llm_eval.models import EvalRun

app = typer.Typer(name="llm-eval", help="LLM Response Evaluation Framework")
console = Console(stderr=True)


@app.command()
def run(
    scenario_path: Path = typer.Argument(..., help="Path to YAML scenario file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running"),
    skip_judge: bool = typer.Option(False, "--skip-judge", help="Skip LLM judge analysis"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="Output directory"),
    provider: list[str] = typer.Option(
        None, "--provider", "-p", help="Only run these providers (can repeat)"
    ),
    judge: str = typer.Option(
        None, "--judge", "-j", help="Override judge model (provider/model, e.g. openai/gpt-4o)"
    ),
) -> None:
    """Run an evaluation scenario."""
    # Load and validate
    scenario = load_scenario(scenario_path)
    settings = get_settings()

    # Override judge if specified
    if judge:
        parts = judge.split("/", 1)
        if len(parts) != 2:
            console.print("[red]--judge must be provider/model (e.g. openai/gpt-4o)[/red]")
            raise typer.Exit(1)
        from llm_eval.models import JudgeSpec
        scenario.judge = JudgeSpec(provider=parts[0], model=parts[1])

    # Filter models to selected providers
    if provider:
        scenario.models = [m for m in scenario.models if m.provider in provider]
        if not scenario.models:
            console.print(f"[red]No models match providers: {provider}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Scenario:[/bold] {scenario.name}")
    console.print(f"  Prompts: {len(scenario.prompts)}")
    console.print(f"  Models: {[f'{m.provider}/{m.model}' for m in scenario.models]}")
    console.print(f"  Runs per prompt: {scenario.runs_per_prompt}")

    languages = set()
    for p in scenario.prompts:
        languages.update(p.variants.keys())
    console.print(f"  Languages: {sorted(languages)}")

    total_calls = (
        len(scenario.prompts)
        * len(languages)
        * len(scenario.models)
        * scenario.runs_per_prompt
    )
    console.print(f"  Total API calls: {total_calls}")

    # Validate API keys
    needed_providers = {m.provider for m in scenario.models}
    if scenario.judge:
        needed_providers.add(scenario.judge.provider)

    missing = []
    if "anthropic" in needed_providers and not settings.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if "openai" in needed_providers and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if "gemini" in needed_providers and not settings.gemini_api_key:
        missing.append("GEMINI_API_KEY")

    if missing and not dry_run:
        console.print(f"[red]Missing API keys: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print("\n[green]Dry run: scenario is valid.[/green]")
        raise typer.Exit(0)

    # Execute

    eval_run = asyncio.run(_run_full(scenario, skip_judge))

    # Save output
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{scenario.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    from llm_eval.output.formats import (
        save_analysis_summary_csv,
        save_responses_csv,
        save_results_json,
    )

    results_path = run_dir / "results.json"
    save_results_json(eval_run, results_path)
    save_responses_csv(eval_run, run_dir / "responses.csv")
    save_analysis_summary_csv(eval_run, run_dir / "analysis_summary.csv")

    from llm_eval.output.charts import generate_keyword_charts

    generate_keyword_charts(eval_run, run_dir)

    console.print(f"\n[bold green]Output saved to {run_dir}/[/bold green]")

    # Print summary
    from llm_eval.output.report import print_summary

    print_summary(eval_run)


async def _run_full(scenario, skip_judge: bool) -> EvalRun:
    """Run execution + analysis pipeline."""
    from llm_eval.analysis import (
        run_deterministic_analysis,
        run_judge_analysis,
        run_statistical_analysis,
    )
    from llm_eval.engine import run_scenario

    # Phase 1: Execute
    eval_run = await run_scenario(scenario)

    # Phase 2: Deterministic analysis
    console.print("\n[bold]Running deterministic analysis...[/bold]")
    eval_run.deterministic = run_deterministic_analysis(eval_run.responses)

    # Phase 3: Statistical analysis
    console.print("[bold]Running statistical analysis...[/bold]")
    eval_run.statistical = run_statistical_analysis(
        eval_run.responses, eval_run.deterministic
    )

    # Phase 4: LLM judge
    if not skip_judge:
        eval_run.judge_results = await run_judge_analysis(eval_run.responses, scenario)

    return eval_run


@app.command()
def analyze(
    results_path: Path = typer.Argument(..., help="Path to results.json from a previous run"),
) -> None:
    """Re-run analysis on existing results."""
    if not results_path.exists():
        console.print(f"[red]File not found: {results_path}[/red]")
        raise typer.Exit(1)

    with open(results_path) as f:
        data = json.load(f)

    eval_run = EvalRun.model_validate(data)
    n = len(eval_run.responses)
    console.print(f"[bold]Loaded {n} responses from {eval_run.scenario_name}[/bold]")

    # Re-run analysis
    from llm_eval.analysis import run_deterministic_analysis, run_statistical_analysis

    eval_run.deterministic = run_deterministic_analysis(eval_run.responses)
    eval_run.statistical = run_statistical_analysis(
        eval_run.responses, eval_run.deterministic
    )

    # Overwrite results
    from llm_eval.output.formats import save_analysis_summary_csv, save_results_json

    save_results_json(eval_run, results_path)
    summary_path = results_path.parent / "analysis_summary.csv"
    save_analysis_summary_csv(eval_run, summary_path)

    console.print(f"[green]Updated {results_path} and {summary_path}[/green]")

    from llm_eval.output.report import print_summary

    print_summary(eval_run)


@app.command("list-providers")
def list_providers_cmd() -> None:
    """Show configured providers and their API key status."""
    from llm_eval.providers import list_providers

    settings = get_settings()
    providers = list_providers()

    from rich.table import Table

    table = Table(title="Configured Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("API Key Status")

    key_map = {
        "anthropic": bool(settings.anthropic_api_key),
        "openai": bool(settings.openai_api_key),
        "gemini": bool(settings.gemini_api_key),
    }

    for p in providers:
        has_key = key_map.get(p, False)
        status = "[green]configured[/green]" if has_key else "[red]missing[/red]"
        table.add_row(p, status)

    Console().print(table)


if __name__ == "__main__":
    app()
