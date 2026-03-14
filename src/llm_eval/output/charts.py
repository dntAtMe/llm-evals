"""Generate keyword charts: bubble chart + per-language column chart."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from llm_eval.models import EvalRun, JudgeResult

matplotlib.use("Agg")
console = Console(stderr=True)

LANG_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

CATEGORY_MARKERS = {
    "varieties": "o",
    "techniques": "s",
    "timing": "D",
    "tools_products": "^",
}

CATEGORY_LABELS = {
    "varieties": "Varieties",
    "techniques": "Techniques",
    "timing": "Timing",
    "tools_products": "Tools & Products",
}

CATEGORY_COLORS = {
    "varieties": "#4C72B0",
    "techniques": "#DD8452",
    "timing": "#55A868",
    "tools_products": "#C44E52",
}


def _collect_categorized_terms(jr: JudgeResult) -> dict[str, list[str]]:
    """Get extracted terms by category."""
    if not jr.extracted_terms:
        return {}
    et = jr.extracted_terms
    result: dict[str, list[str]] = {}
    for cat in ("varieties", "techniques", "timing", "tools_products"):
        terms = getattr(et, cat)
        if terms:
            result[cat] = [t.lower().strip() for t in terms]
    return result


def _build_term_data(
    results: list[JudgeResult],
    languages: list[str],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, tuple[str, set[str]]]]:
    """Shared data prep: per-language categorized terms + unified term info."""
    lang_cat_terms: dict[str, dict[str, list[str]]] = {}
    for jr in results:
        lang_cat_terms[jr.language] = _collect_categorized_terms(jr)

    # term -> (category, set of languages)
    term_info: dict[str, tuple[str, set[str]]] = {}
    for lang in languages:
        for cat, terms in lang_cat_terms.get(lang, {}).items():
            for term in terms:
                if term not in term_info:
                    term_info[term] = (cat, set())
                term_info[term][1].add(lang)

    return lang_cat_terms, term_info


def _lang_flat_terms(lang_cat_terms: dict[str, list[str]]) -> set[str]:
    """Flatten all categories into a single set."""
    result: set[str] = set()
    for terms in lang_cat_terms.values():
        result.update(terms)
    return result


# ---------------------------------------------------------------------------
# Chart 1: Bubble chart (terms x languages, shape = category)
# ---------------------------------------------------------------------------


def _draw_bubble_chart(
    ax: plt.Axes,
    sorted_terms: list[str],
    term_info: dict[str, tuple[str, set[str]]],
    lang_cat_terms: dict[str, dict[str, list[str]]],
    languages: list[str],
    color_map: dict[str, str],
    title: str,
) -> None:
    n_langs = len(languages)
    bubble_size = 320

    for xi, lang in enumerate(languages):
        all_lang_terms = _lang_flat_terms(lang_cat_terms.get(lang, {}))
        for yi, term in enumerate(sorted_terms):
            cat = term_info[term][0]
            marker = CATEGORY_MARKERS.get(cat, "o")
            present = term in all_lang_terms

            if present:
                ax.scatter(
                    xi, yi, s=bubble_size, c=color_map[lang], marker=marker,
                    alpha=0.8, edgecolors="white", linewidths=0.8, zorder=3,
                )
            else:
                ax.scatter(
                    xi, yi, s=bubble_size * 0.3, c="none", marker=marker,
                    edgecolors="#d0d0d0", linewidths=0.8, alpha=0.4, zorder=2,
                )

    ax.set_xticks(range(n_langs))
    ax.set_xticklabels([l.upper() for l in languages], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(sorted_terms)))
    ax.set_yticklabels(sorted_terms, fontsize=9)
    ax.set_xlim(-0.5, n_langs - 0.5)
    ax.set_ylim(len(sorted_terms) - 0.5, -0.5)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", alpha=0.1)
    for xi in range(n_langs):
        ax.axvline(xi, color="#e0e0e0", linewidth=0.5, alpha=0.5, zorder=1)

    # Shared / unique annotations
    for yi, term in enumerate(sorted_terms):
        n_present = len(term_info[term][1])
        if n_present == n_langs:
            ax.annotate(
                "shared", (n_langs - 0.5, yi), xytext=(8, 0),
                textcoords="offset points", fontsize=7, color="green", va="center",
            )
        elif n_present == 1:
            ax.annotate(
                "unique", (n_langs - 0.5, yi), xytext=(8, 0),
                textcoords="offset points", fontsize=7, color="#999999", va="center",
            )

    # Category legend
    legend_handles = []
    cats_present = {term_info[t][0] for t in sorted_terms}
    for cat in ("varieties", "techniques", "timing", "tools_products"):
        if cat not in cats_present:
            continue
        handle = ax.scatter(
            [], [], s=80, c="#666666", marker=CATEGORY_MARKERS[cat],
            label=CATEGORY_LABELS[cat],
        )
        legend_handles.append(handle)

    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=8,
        title="Category", title_fontsize=9, framealpha=0.9,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)


# ---------------------------------------------------------------------------
# Chart 2: Column chart (one column per language, terms listed by category)
# ---------------------------------------------------------------------------


def _draw_column_chart(
    fig: plt.Figure,
    lang_cat_terms: dict[str, dict[str, list[str]]],
    term_info: dict[str, tuple[str, set[str]]],
    languages: list[str],
    color_map: dict[str, str],
    title: str,
) -> None:
    n_langs = len(languages)
    axes = fig.subplots(1, n_langs, sharey=False)
    if n_langs == 1:
        axes = [axes]

    # Find all terms present across all languages to mark shared ones
    all_terms_all_langs: set[str] = set()
    for cat_terms in lang_cat_terms.values():
        all_terms_all_langs |= _lang_flat_terms(cat_terms)

    shared_terms = {
        t for t, (_, langs) in term_info.items() if len(langs) == n_langs
    }

    max_rows = 0
    for ax_idx, lang in enumerate(languages):
        ax = axes[ax_idx]
        cat_terms = lang_cat_terms.get(lang, {})

        # Build rows: [(term, category), ...] grouped by category
        rows: list[tuple[str, str]] = []
        for cat in ("varieties", "techniques", "timing", "tools_products"):
            terms = cat_terms.get(cat, [])
            if terms:
                # Add category header
                rows.append((f"— {CATEGORY_LABELS[cat]} —", "_header"))
                for t in sorted(terms):
                    rows.append((t, cat))

        max_rows = max(max_rows, len(rows))

        # Draw as text list
        ax.set_xlim(0, 1)
        ax.set_ylim(len(rows) + 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        for yi, (term, cat) in enumerate(rows):
            if cat == "_header":
                ax.text(
                    0.05, yi, term, fontsize=8, fontweight="bold",
                    color="#555555", va="center",
                )
            else:
                is_shared = term in shared_terms
                color = CATEGORY_COLORS.get(cat, "#333333")
                # Shared terms get a green dot prefix
                prefix = "\u25cf " if is_shared else "  "
                ax.text(
                    0.08, yi, prefix + term, fontsize=9, va="center",
                    color=color if not is_shared else "#2d8a4e",
                )

        # Column header
        ax.set_title(
            lang.upper(), fontsize=13, fontweight="bold",
            color=color_map[lang], pad=10,
        )

    # Normalize y-limits so columns align
    for ax in axes:
        ax.set_ylim(max_rows + 0.5, -0.5)

    # Add a small legend for shared marker
    fig.text(
        0.5, 0.01,
        "\u25cf green = shared across all languages",
        ha="center", fontsize=8, color="#2d8a4e",
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_keyword_charts(run: EvalRun, output_dir: Path) -> list[Path]:
    """Generate bubble + column keyword charts, one pair per (prompt_id, model)."""
    if not run.judge_results:
        return []

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Group judge results by (prompt_id, provider, model)
    groups: dict[tuple[str, str, str], list[JudgeResult]] = defaultdict(list)
    for jr in run.judge_results:
        groups[(jr.prompt_id, jr.provider, jr.model)].append(jr)

    saved: list[Path] = []

    for (prompt_id, provider, model), results in groups.items():
        languages = sorted({jr.language for jr in results})
        if len(languages) < 2:
            continue

        lang_cat_terms, term_info = _build_term_data(results, languages)
        if not term_info:
            continue

        color_map = {
            lang: LANG_COLORS[i % len(LANG_COLORS)]
            for i, lang in enumerate(languages)
        }
        safe_model = model.replace("/", "-")
        subtitle = f"{prompt_id} — {provider}/{model}"

        # --- Bubble chart ---
        sorted_terms = sorted(
            term_info.keys(),
            key=lambda t: (-len(term_info[t][1]), term_info[t][0], t),
        )[:35]

        n_terms = len(sorted_terms)
        fig_b, ax_b = plt.subplots(
            figsize=(3 + len(languages) * 1.5, max(5, n_terms * 0.38))
        )
        _draw_bubble_chart(
            ax_b, sorted_terms, term_info, lang_cat_terms,
            languages, color_map, f"Extracted Terms — {subtitle}",
        )
        plt.tight_layout()
        bubble_path = charts_dir / f"{prompt_id}_{provider}_{safe_model}_bubble.png"
        fig_b.savefig(bubble_path, dpi=150, bbox_inches="tight")
        plt.close(fig_b)
        saved.append(bubble_path)

        # --- Column chart ---
        max_terms = max(
            (sum(len(ts) for ts in ct.values()) + len(ct)
             for ct in lang_cat_terms.values()),
            default=5,
        )
        fig_c = plt.figure(
            figsize=(3.5 * len(languages), max(4, max_terms * 0.35))
        )
        _draw_column_chart(
            fig_c, lang_cat_terms, term_info, languages, color_map,
            f"Terms by Language — {subtitle}",
        )
        col_path = charts_dir / f"{prompt_id}_{provider}_{safe_model}_columns.png"
        fig_c.savefig(col_path, dpi=150, bbox_inches="tight")
        plt.close(fig_c)
        saved.append(col_path)

    if saved:
        console.print(f"[dim]Saved {len(saved)} keyword charts to {charts_dir}/[/dim]")

    return saved
