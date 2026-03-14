"""JSON and CSV serialization for evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_eval.models import EvalRun


def save_results_json(run: EvalRun, path: Path) -> None:
    """Save full structured results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = run.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_responses_csv(run: EvalRun, path: Path) -> None:
    """Save one row per response to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id", "language", "provider", "model", "run_index",
        "response_length", "latency_ms", "input_tokens", "output_tokens",
        "response_text",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in run.responses:
            writer.writerow({
                "prompt_id": r.prompt_id,
                "language": r.language,
                "provider": r.provider,
                "model": r.model,
                "run_index": r.run_index,
                "response_length": len(r.response_text),
                "latency_ms": round(r.latency_ms, 1),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "response_text": r.response_text,
            })


def save_analysis_summary_csv(run: EvalRun, path: Path) -> None:
    """Save one row per (prompt, model) with key analysis metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Detect languages from responses
    languages = sorted({r.language for r in run.responses})

    fieldnames = ["prompt_id", "provider", "model", "cross_lang_similarity"]
    for lang in languages:
        fieldnames.append(f"within_lang_{lang}")
    for lang in languages:
        fieldnames.extend([f"len_mean_{lang}", f"len_cv_{lang}"])
    fieldnames.extend(["cross_lang_divergence", "gap_ratio"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        stat_index = {
            (s.prompt_id, s.provider, s.model): s for s in run.statistical
        }

        for det in run.deterministic:
            stat = stat_index.get((det.prompt_id, det.provider, det.model))
            row: dict = {
                "prompt_id": det.prompt_id,
                "provider": det.provider,
                "model": det.model,
            }
            if det.cross_language_similarity:
                vals = list(det.cross_language_similarity.values())
                row["cross_lang_similarity"] = round(sum(vals) / len(vals), 4)
            if det.within_language_similarity:
                for lang in languages:
                    row[f"within_lang_{lang}"] = det.within_language_similarity.get(lang, "")
            if det.length_stats:
                for lang in languages:
                    stats = det.length_stats.get(lang)
                    if stats:
                        row[f"len_mean_{lang}"] = stats.mean
                        row[f"len_cv_{lang}"] = stats.cv
            if stat:
                row["cross_lang_divergence"] = stat.cross_language_divergence
                row["gap_ratio"] = stat.gap_ratio

            writer.writerow(row)
