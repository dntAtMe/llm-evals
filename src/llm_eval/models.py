"""Pydantic models for scenarios, responses, and evaluation results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# --- Scenario models (parsed from YAML) ---


class PromptVariant(BaseModel):
    """A single prompt with language variants."""

    id: str
    variants: dict[str, str]  # language code -> prompt text


class ModelSpec(BaseModel):
    """Which provider + model to query."""

    provider: str
    model: str


class JudgeSpec(BaseModel):
    """Which model to use as the LLM judge."""

    provider: str
    model: str


class Scenario(BaseModel):
    """A complete evaluation scenario loaded from YAML."""

    name: str
    prompts: list[PromptVariant]
    models: list[ModelSpec]
    runs_per_prompt: int = 5
    judge: JudgeSpec | None = None


# --- Runtime models ---


class LLMResponse(BaseModel):
    """A single response from an LLM."""

    prompt_id: str
    language: str
    provider: str
    model: str
    run_index: int
    prompt_text: str
    response_text: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- Analysis models ---


class LengthStats(BaseModel):
    mean: float
    std: float
    cv: float  # coefficient of variation


class KeywordResult(BaseModel):
    language: str
    keywords: list[str]


class DeterministicResult(BaseModel):
    """Results from deterministic analysis for a (prompt, model) pair."""

    prompt_id: str
    provider: str
    model: str
    cross_language_similarity: dict[str, float] | None = None  # lang_pair -> cosine sim
    within_language_similarity: dict[str, float] | None = None  # language -> mean pairwise sim
    length_stats: dict[str, LengthStats] | None = None  # language -> stats
    keywords: list[KeywordResult] | None = None


class JudgeScore(BaseModel):
    specificity: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)


class ExtractedTerms(BaseModel):
    """Key terms extracted from a response, translated to English."""

    varieties: list[str] = []  # plant varieties / cultivars
    techniques: list[str] = []  # gardening methods / practices
    timing: list[str] = []  # planting dates, seasons, calendar refs
    tools_products: list[str] = []  # tools, fertilizers, products mentioned


class JudgeResult(BaseModel):
    """LLM judge evaluation for a (prompt, model, language) group."""

    prompt_id: str
    provider: str
    model: str
    language: str
    quality_scores: JudgeScore | None = None
    extracted_terms: ExtractedTerms | None = None


class StatisticalResult(BaseModel):
    """Statistical analysis for a (prompt, model) pair."""

    prompt_id: str
    provider: str
    model: str
    cross_run_variance: dict[str, float] | None = None  # language -> mean pairwise sim
    cross_language_divergence: float | None = None  # centroid similarity
    gap_ratio: float | None = None  # inter / intra


# --- Top-level run result ---


class EvalRun(BaseModel):
    """Complete evaluation run output."""

    scenario_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    responses: list[LLMResponse]
    deterministic: list[DeterministicResult] = []
    judge_results: list[JudgeResult] = []
    statistical: list[StatisticalResult] = []
    metadata: dict[str, Any] = {}
