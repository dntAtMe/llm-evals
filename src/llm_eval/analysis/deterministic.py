"""Deterministic analysis: embeddings similarity, length stats, keywords."""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

import numpy as np
from rich.console import Console

from llm_eval.models import (
    DeterministicResult,
    KeywordResult,
    LengthStats,
    LLMResponse,
)

console = Console(stderr=True)

# Lazy-loaded model
_embed_model = None

# Stopwords for keyword extraction
EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "about", "up", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "our", "their", "what", "which",
    "who", "whom", "also", "make", "like", "well", "much", "many", "get",
    "use", "one", "two", "new", "way", "help", "keep", "good",
}

PL_STOPWORDS = {
    "i", "w", "na", "z", "do", "to", "nie", "się", "jest", "że", "o",
    "co", "jak", "ale", "za", "od", "po", "tak", "już", "ich", "czy",
    "ten", "ta", "te", "też", "go", "je", "są", "lub", "dla", "przez",
    "przy", "aby", "oraz", "być", "jako", "jej", "jego", "tym", "tego",
    "tej", "które", "który", "która", "których", "którzy", "można",
    "np", "tzw", "bardzo", "tylko", "między", "przed", "nad", "pod",
    "bez", "ze", "we", "gdy", "kiedy", "gdzie", "będzie", "był",
    "była", "było", "więc", "bo", "sobie", "mu", "mi", "mnie",
    "nam", "nas", "ich", "im", "je", "ją",
}

SV_STOPWORDS = {
    "och", "att", "det", "som", "den", "för", "med", "har", "inte",
    "ett", "kan", "till", "vara", "från", "eller", "men", "ska",
    "också", "vid", "sin", "alla", "sig", "här", "där", "man",
    "bli", "blir", "blev", "denna", "dessa", "när", "hur", "vad",
    "vilka", "mycket", "efter", "upp", "sedan", "bara", "andra",
    "mer", "under", "över", "dig", "mig", "oss", "dem", "dom",
    "jag", "han", "hon", "det", "den", "dem", "sitt", "sin",
}

TH_STOPWORDS = {
    "ที่", "และ", "ใน", "ของ", "ได้", "ไม่", "จะ", "เป็น", "มี",
    "กับ", "ให้", "แต่", "หรือ", "อยู่", "คือ", "จาก", "ด้วย",
    "ก็", "แล้ว", "ไป", "มา", "นี้", "ซึ่ง", "โดย", "เมื่อ",
    "ถ้า", "อีก", "ยัง", "ทั้ง", "เช่น", "ต้อง", "ว่า", "กัน",
    "ทำ", "จึง", "ถึง", "ครับ", "ค่ะ", "นั้น", "บ้าง", "อย่าง",
}

JA_STOPWORDS = {
    # Particles
    "は", "が", "を", "に", "で", "と", "の", "も", "や", "か", "へ", "より",
    "から", "まで", "として", "について", "によって", "において", "に対して",
    # Auxiliary verbs / copula
    "です", "ます", "ある", "いる", "する", "なる", "れる", "られる", "せる",
    "させる", "ない", "ません", "でした", "ました", "だ", "た", "て", "で",
    # Conjunctions / connectives
    "そして", "また", "しかし", "ただし", "なお", "さらに", "つまり", "例えば",
    "ので", "けど", "が", "ば", "たら", "ながら", "ため", "ように", "ことが",
    # Common short words / pronouns
    "この", "その", "あの", "どの", "これ", "それ", "あれ", "どれ", "ここ",
    "そこ", "あそこ", "私", "あなた", "彼", "彼女", "私たち", "など", "等",
    "こと", "もの", "とき", "ところ", "よう", "ほう", "方", "時", "中",
    # Numbers / counters commonly used as filler
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
}

STOPWORDS = {
    "en": EN_STOPWORDS,
    "pl": PL_STOPWORDS,
    "sv": SV_STOPWORDS,
    "th": TH_STOPWORDS,
    "ja": JA_STOPWORDS,
}


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        console.print("[dim]Loading multilingual embedding model...[/dim]")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embed_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _pairwise_mean_similarity(embeddings: list[np.ndarray]) -> float:
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for a, b in combinations(range(len(embeddings)), 2):
        sims.append(_cosine_similarity(embeddings[a], embeddings[b]))
    return float(np.mean(sims))


_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]{2,}")
_LATIN_PATTERN = re.compile(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}")


def _extract_keywords(texts: list[str], language: str, top_n: int = 15) -> list[str]:
    """Extract top keywords by term frequency, filtering stopwords."""
    stopwords = STOPWORDS.get(language, set())
    word_counts: dict[str, int] = defaultdict(int)
    for text in texts:
        # CJK languages: extract character sequences (no lowercasing — case doesn't apply)
        cjk_words = _CJK_PATTERN.findall(text)
        for w in cjk_words:
            if w not in stopwords:
                word_counts[w] += 1
        # Latin-script words (covers EN, PL, SV, romanized terms mixed into CJK text)
        latin_words = _LATIN_PATTERN.findall(text.lower())
        for w in latin_words:
            if w not in stopwords:
                word_counts[w] += 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def _compute_length_stats(texts: list[str]) -> LengthStats:
    lengths = [len(t) for t in texts]
    mean = float(np.mean(lengths))
    std = float(np.std(lengths))
    cv = std / mean if mean > 0 else 0.0
    return LengthStats(mean=round(mean, 1), std=round(std, 1), cv=round(cv, 3))


def run_deterministic_analysis(
    responses: list[LLMResponse],
) -> list[DeterministicResult]:
    """Run deterministic analysis on all responses, grouped by (prompt_id, provider, model)."""
    model = _get_embed_model()

    # Group responses
    groups: dict[tuple[str, str, str], list[LLMResponse]] = defaultdict(list)
    for r in responses:
        groups[(r.prompt_id, r.provider, r.model)].append(r)

    results: list[DeterministicResult] = []

    for (prompt_id, provider, model_name), group_responses in groups.items():
        # Sub-group by language
        by_lang: dict[str, list[LLMResponse]] = defaultdict(list)
        for r in group_responses:
            by_lang[r.language].append(r)

        languages = sorted(by_lang.keys())

        # Compute embeddings per language
        lang_embeddings: dict[str, list[np.ndarray]] = {}
        for lang in languages:
            texts = [r.response_text for r in by_lang[lang]]
            embeddings = model.encode(texts, convert_to_numpy=True)
            lang_embeddings[lang] = list(embeddings)

        # Within-language similarity
        within_sim: dict[str, float] = {}
        for lang in languages:
            within_sim[lang] = round(_pairwise_mean_similarity(lang_embeddings[lang]), 4)

        # Cross-language similarity (centroid-based)
        cross_sim: dict[str, float] = {}
        for la, lb in combinations(languages, 2):
            centroid_a = np.mean(lang_embeddings[la], axis=0)
            centroid_b = np.mean(lang_embeddings[lb], axis=0)
            key = f"{la}-{lb}"
            cross_sim[key] = round(_cosine_similarity(centroid_a, centroid_b), 4)

        # Length stats
        length_stats: dict[str, LengthStats] = {}
        for lang in languages:
            texts = [r.response_text for r in by_lang[lang]]
            length_stats[lang] = _compute_length_stats(texts)

        # Keywords
        keywords: list[KeywordResult] = []
        for lang in languages:
            texts = [r.response_text for r in by_lang[lang]]
            kws = _extract_keywords(texts, lang)
            keywords.append(KeywordResult(language=lang, keywords=kws))

        results.append(
            DeterministicResult(
                prompt_id=prompt_id,
                provider=provider,
                model=model_name,
                cross_language_similarity=cross_sim,
                within_language_similarity=within_sim,
                length_stats=length_stats,
                keywords=keywords,
            )
        )

    return results
