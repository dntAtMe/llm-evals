"""Rich CLI tables summary for evaluation results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from llm_eval.models import CrossLanguageComparison, EvalRun

console = Console()


def _print_cross_language_table(title: str, comparisons: list[CrossLanguageComparison]) -> None:
    """Print one Rich table for a list of cross-language comparison results."""
    if not comparisons:
        return
    table = Table(title=title, show_lines=True)
    table.add_column("Prompt", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Category")
    table.add_column("Difference", max_width=60)
    table.add_column("Languages")

    for cl in comparisons:
        model_str = f"{cl.provider}/{cl.model}"
        if not cl.differences:
            table.add_row(
                cl.prompt_id, model_str, "", "[dim]no differences found[/dim]", ""
            )
            continue
        for i, diff in enumerate(cl.differences):
            cat_colors = {
                "recommendations": "yellow",
                "assumptions": "red",
                "omissions": "magenta",
                "emphasis": "blue",
                "depth": "cyan",
            }
            color = cat_colors.get(diff.category, "white")
            cat = f"[{color}]{diff.category}[/{color}]"
            langs = ", ".join(diff.languages_affected)
            table.add_row(
                cl.prompt_id if i == 0 else "",
                model_str if i == 0 else "",
                cat,
                diff.description,
                langs,
            )

    console.print(table)

    for cl in comparisons:
        if cl.summary:
            mode = cl.aggregation_mode or "legacy"
            console.print(
                f"  [dim]{cl.prompt_id} ({cl.provider}/{cl.model}) [{mode}]:[/dim] "
                f"{cl.summary}"
            )
    console.print()


def _detect_languages(run: EvalRun) -> list[str]:
    """Detect languages present in the run, sorted."""
    langs: set[str] = set()
    for r in run.responses:
        langs.add(r.language)
    return sorted(langs)


def print_summary(run: EvalRun) -> None:
    """Print a rich summary table of the evaluation run."""
    console.print(f"\n[bold green]Evaluation: {run.scenario_name}[/bold green]")
    console.print(f"Responses collected: {len(run.responses)}\n")

    languages = _detect_languages(run)

    # --- Deterministic metrics table ---
    if run.deterministic:
        table = Table(title="Deterministic Analysis", show_lines=True)
        table.add_column("Prompt", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Cross-lang\nsimilarity")
        for lang in languages:
            table.add_column(f"Within {lang.upper()}\nstability")
        for lang in languages:
            table.add_column(f"Len {lang.upper()}\n(mean/CV)")

        for det in run.deterministic:
            row: list[str] = [det.prompt_id, f"{det.provider}/{det.model}"]

            # Cross-language: show mean across all pairs
            if det.cross_language_similarity:
                vals = list(det.cross_language_similarity.values())
                row.append(f"{sum(vals) / len(vals):.4f}")
            else:
                row.append("")

            # Within-language stability
            for lang in languages:
                if det.within_language_similarity and lang in det.within_language_similarity:
                    row.append(f"{det.within_language_similarity[lang]:.4f}")
                else:
                    row.append("")

            # Length stats
            for lang in languages:
                if det.length_stats and lang in det.length_stats:
                    s = det.length_stats[lang]
                    row.append(f"{s.mean:.0f} / {s.cv:.3f}")
                else:
                    row.append("")

            table.add_row(*row)

        console.print(table)

        # Cross-language similarity detail (per pair)
        if languages and len(languages) > 2 and run.deterministic[0].cross_language_similarity:
            table = Table(title="Cross-language Similarity (per pair)", show_lines=True)
            table.add_column("Prompt", style="cyan")
            table.add_column("Model", style="magenta")
            pairs = sorted(run.deterministic[0].cross_language_similarity.keys())
            for pair in pairs:
                table.add_column(pair.upper())

            for det in run.deterministic:
                row = [det.prompt_id, f"{det.provider}/{det.model}"]
                for pair in pairs:
                    val = (det.cross_language_similarity or {}).get(pair)
                    row.append(f"{val:.4f}" if val is not None else "")
                table.add_row(*row)

            console.print(table)

    # --- Statistical metrics table ---
    if run.statistical:
        table = Table(title="Statistical Analysis", show_lines=True)
        table.add_column("Prompt", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Cross-lang\ndivergence")
        table.add_column("Gap ratio\n(>1 = lang matters)")

        for stat in run.statistical:
            div = f"{stat.cross_language_divergence:.4f}" if stat.cross_language_divergence else ""
            gap = ""
            if stat.gap_ratio is not None:
                color = "red" if stat.gap_ratio > 1 else "green"
                gap = f"[{color}]{stat.gap_ratio:.4f}[/{color}]"

            table.add_row(
                stat.prompt_id,
                f"{stat.provider}/{stat.model}",
                div, gap,
            )

        console.print(table)

    # --- Judge results table ---
    if run.judge_results:
        table = Table(title="LLM Judge — Quality & Extracted Terms", show_lines=True)
        table.add_column("Prompt", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Lang")
        table.add_column("Quality\n(scores)")
        table.add_column("Plants (EN)")
        table.add_column("Techniques (EN)")
        table.add_column("Timing (EN)")

        for jr in run.judge_results:
            quality = ""
            if jr.quality_scores:
                quality = " ".join(
                    f"{k[:4]}:{v}" for k, v in jr.quality_scores.scores.items()
                )

            et = jr.extracted_terms
            plants = ", ".join(et.plants[:6]) if et else ""
            techniques = ", ".join(et.techniques[:4]) if et else ""
            timing = ", ".join(et.timing[:4]) if et else ""

            table.add_row(
                jr.prompt_id,
                f"{jr.provider}/{jr.model}",
                jr.language,
                quality,
                plants,
                techniques,
                timing,
            )

        console.print(table)

    # --- Cross-language semantic differences (concat / summarized / legacy) ---
    if run.cross_language:
        concat_cl = [c for c in run.cross_language if c.aggregation_mode == "concatenated"]
        summ_cl = [c for c in run.cross_language if c.aggregation_mode == "summarized"]
        legacy_cl = [c for c in run.cross_language if c.aggregation_mode is None]

        _print_cross_language_table(
            "Cross-language Semantic Differences (concatenated runs)",
            concat_cl,
        )
        _print_cross_language_table(
            "Cross-language Semantic Differences (from per-language summaries)",
            summ_cl,
        )
        _print_cross_language_table(
            "Cross-language Semantic Differences (legacy)",
            legacy_cl,
        )

    # --- Keywords summary ---
    if run.deterministic:
        table = Table(title="Top Keywords by Language", show_lines=True)
        table.add_column("Prompt", style="cyan")
        table.add_column("Model", style="magenta")
        for lang in languages:
            table.add_column(f"{lang.upper()} keywords")

        for det in run.deterministic:
            row = [det.prompt_id, f"{det.provider}/{det.model}"]
            kw_by_lang: dict[str, str] = {}
            if det.keywords:
                for kw in det.keywords:
                    kw_by_lang[kw.language] = ", ".join(kw.keywords[:10])
            for lang in languages:
                row.append(kw_by_lang.get(lang, ""))
            table.add_row(*row)

        console.print(table)
