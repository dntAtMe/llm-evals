"""Unit tests for cross-language aggregation helpers (no API calls)."""

from llm_eval.analysis.llm_judge import _build_concat_responses_block
from llm_eval.models import LLMResponse


def _r(lang: str, run_index: int, text: str) -> LLMResponse:
    return LLMResponse(
        prompt_id="p",
        language=lang,
        provider="x",
        model="m",
        run_index=run_index,
        prompt_text="question",
        response_text=text,
        latency_ms=0.0,
    )


def test_build_concat_sorts_runs_and_languages():
    by_lang = {
        "en": [_r("en", 1, "second"), _r("en", 0, "first")],
        "ja": [_r("ja", 0, "only")],
    }
    block = _build_concat_responses_block(by_lang)
    assert "**Run 1:**\nfirst" in block
    assert "**Run 2:**\nsecond" in block
    assert block.index("EN") < block.index("JA")
    assert "**Summary" not in block
