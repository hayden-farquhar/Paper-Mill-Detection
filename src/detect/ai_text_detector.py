"""
Detect markers of LLM-generated text in academic papers.

Uses lightweight, model-free heuristics suitable for CPU-only environments:
- Vocabulary diversity (type-token ratio)
- Sentence length uniformity (burstiness)
- Hapax legomena ratio (words appearing exactly once)
- Connective/discourse marker density
- Repetitive phrase patterns

These are not definitive AI-text detectors but provide useful signal
features for the ensemble classifier, especially post-ChatGPT (2023+).
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Discourse markers overused by LLMs
LLM_DISCOURSE_MARKERS = [
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bnotably\b",
    r"\bspecifically\b",
    r"\bimportantly\b",
    r"\badditionally\b",
    r"\bin particular\b",
    r"\bit is worth noting\b",
    r"\bit should be noted\b",
    r"\bit is important to note\b",
    r"\boverall\b",
    r"\bin summary\b",
    r"\bconversely\b",
    r"\bnevertheless\b",
    r"\bnonetheless\b",
    r"\bin contrast\b",
    r"\bsignificantly\b",
    r"\bsubsequently\b",
    r"\bcorrespondingly\b",
    r"\binterestingly\b",
    r"\bremarkably\b",
]

# Phrases that are characteristically LLM-generated
LLM_CHARACTERISTIC_PHRASES = [
    r"\bdelve into\b",
    r"\btapestry of\b",
    r"\bmultifaceted\b",
    r"\blocated at the intersection of\b",
    r"\bit'?s important to acknowledge\b",
    r"\bunderscores the importance\b",
    r"\bpivotal role\b",
    r"\blandscape of\b",
    r"\bparadigm shift\b",
    r"\bin the realm of\b",
    r"\bholistic approach\b",
    r"\bunprecedented\b",
    r"\bgroundbreaking\b",
    r"\bcomprehensive overview\b",
    r"\brobust framework\b",
    r"\bcritical juncture\b",
    r"\bnavigate the complex\b",
    r"\bshed light on\b",
    r"\bpaves the way\b",
    r"\bremains to be seen\b",
]


@dataclass
class AITextFeatures:
    """Features indicating possible LLM-generated text."""

    vocabulary_diversity: float = 0.0  # type-token ratio (lower = more repetitive)
    hapax_ratio: float = 0.0  # fraction of words appearing exactly once
    sentence_length_uniformity: float = 0.0  # 1 - CV (higher = more uniform = more AI-like)
    mean_sentence_length: float = 0.0
    discourse_marker_density: float = 0.0  # per 1000 words
    llm_phrase_count: int = 0
    llm_phrase_density: float = 0.0  # per 1000 words
    avg_word_length: float = 0.0
    long_word_ratio: float = 0.0  # fraction of words > 10 chars (LLMs use more complex vocab)

    def to_dict(self) -> dict:
        return {
            "ai_vocabulary_diversity": round(self.vocabulary_diversity, 4),
            "ai_hapax_ratio": round(self.hapax_ratio, 4),
            "ai_sentence_uniformity": round(self.sentence_length_uniformity, 4),
            "ai_mean_sentence_length": round(self.mean_sentence_length, 2),
            "ai_discourse_marker_density": round(self.discourse_marker_density, 4),
            "ai_llm_phrase_count": self.llm_phrase_count,
            "ai_llm_phrase_density": round(self.llm_phrase_density, 4),
            "ai_avg_word_length": round(self.avg_word_length, 2),
            "ai_long_word_ratio": round(self.long_word_ratio, 4),
        }


def _tokenize_simple(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def detect_ai_text(text: str) -> AITextFeatures:
    """Compute AI-text detection features for a document.

    Args:
        text: Document text to analyse

    Returns:
        AITextFeatures with all computed metrics
    """
    if not text or len(text) < 100:
        return AITextFeatures()

    features = AITextFeatures()
    tokens = _tokenize_simple(text)

    if not tokens:
        return features

    word_count = len(tokens)

    # Vocabulary diversity (type-token ratio, computed on first 1000 tokens
    # to normalise for document length)
    sample = tokens[:1000]
    features.vocabulary_diversity = len(set(sample)) / len(sample)

    # Hapax legomena ratio
    word_counts = Counter(tokens)
    hapax = sum(1 for c in word_counts.values() if c == 1)
    features.hapax_ratio = hapax / len(word_counts) if word_counts else 0.0

    # Sentence length uniformity
    sentences = _split_sentences(text)
    if len(sentences) >= 3:
        sent_lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(sent_lengths)
        std_len = np.std(sent_lengths)
        cv = std_len / mean_len if mean_len > 0 else 0.0
        features.sentence_length_uniformity = max(0, 1 - cv)
        features.mean_sentence_length = mean_len

    # Discourse marker density
    dm_count = 0
    for pattern in LLM_DISCOURSE_MARKERS:
        dm_count += len(re.findall(pattern, text, re.IGNORECASE))
    features.discourse_marker_density = (dm_count / word_count) * 1000

    # LLM characteristic phrases
    llm_count = 0
    for pattern in LLM_CHARACTERISTIC_PHRASES:
        llm_count += len(re.findall(pattern, text, re.IGNORECASE))
    features.llm_phrase_count = llm_count
    features.llm_phrase_density = (llm_count / word_count) * 1000

    # Word length features
    word_lengths = [len(w) for w in tokens]
    features.avg_word_length = np.mean(word_lengths)
    features.long_word_ratio = sum(1 for l in word_lengths if l > 10) / word_count

    return features


def detect_ai_text_batch(texts: list[str], ids: list[str] = None) -> list[dict]:
    """Compute AI text features for multiple documents."""
    results = []
    for i, text in enumerate(texts):
        feat = detect_ai_text(text or "")
        record = feat.to_dict()
        if ids:
            record["id"] = ids[i]
        results.append(record)
    return results
