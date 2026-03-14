"""Statistical analysis: cross-run variance, cross-language divergence, gap ratio."""

from __future__ import annotations

import numpy as np
from rich.console import Console

from llm_eval.models import DeterministicResult, LLMResponse, StatisticalResult

console = Console(stderr=True)


def run_statistical_analysis(
    responses: list[LLMResponse],
    deterministic_results: list[DeterministicResult],
) -> list[StatisticalResult]:
    """Compute statistical metrics from deterministic analysis results."""
    results: list[StatisticalResult] = []

    for det in deterministic_results:
        # Cross-run variance: how stable are responses within each language?
        cross_run_var = det.within_language_similarity

        # Cross-language divergence: centroid similarity between language pairs
        cross_lang_div: float | None = None
        if det.cross_language_similarity:
            sims = list(det.cross_language_similarity.values())
            cross_lang_div = round(float(np.mean(sims)), 4) if sims else None

        # Gap ratio: inter-language distance / intra-language variance
        # >1 means language matters more than randomness
        gap_ratio: float | None = None
        if cross_lang_div is not None and cross_run_var:
            intra_values = list(cross_run_var.values())
            mean_intra = float(np.mean(intra_values))
            if mean_intra > 0:
                # Convert similarities to distances for the ratio
                inter_distance = 1.0 - cross_lang_div
                intra_distance = 1.0 - mean_intra
                if intra_distance > 0:
                    gap_ratio = round(inter_distance / intra_distance, 4)

        results.append(
            StatisticalResult(
                prompt_id=det.prompt_id,
                provider=det.provider,
                model=det.model,
                cross_run_variance=cross_run_var,
                cross_language_divergence=cross_lang_div,
                gap_ratio=gap_ratio,
            )
        )

    return results
