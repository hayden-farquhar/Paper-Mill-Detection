"""
Score papers for formulaic/template-like structure.

Paper mills often produce papers from templates, resulting in:
- Identical section ordering (Introduction/Methods/Results/Discussion)
- Boilerplate methods sections with recycled text
- Similar abstract structures and sentence patterns
- Unusually uniform paragraph and sentence lengths

This module quantifies these structural features.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical section headings (case-insensitive matching)
STANDARD_SECTIONS = [
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
]

# Common aliases mapped to canonical names
SECTION_ALIASES = {
    "background": "introduction",
    "materials and methods": "methods",
    "methodology": "methods",
    "experimental": "methods",
    "study design": "methods",
    "statistical analysis": "methods",
    "findings": "results",
    "experimental results": "results",
    "outcomes": "results",
    "limitations": "discussion",
    "conclusions": "conclusion",
    "summary": "conclusion",
}

# Boilerplate sentence patterns common in paper mill output
BOILERPLATE_PATTERNS = [
    r"in this study,? we (?:propose|present|develop|introduce|investigate)",
    r"the (?:results|findings) (?:show|indicate|demonstrate|suggest|reveal) that",
    r"to the best of our knowledge",
    r"in recent years,? .{10,60} has (?:attracted|received|gained) .{5,30} attention",
    r"(?:plays|play) an? (?:important|crucial|vital|significant|key|essential) role",
    r"has been widely (?:used|applied|adopted|studied|investigated)",
    r"with the (?:rapid|fast) development of",
    r"has (?:become|emerged as) (?:a |an )?(?:hot|popular|important|promising)",
    r"(?:provides|provided|offering) a (?:new|novel|promising|effective) (?:approach|method|framework|strategy)",
    r"(?:however|nevertheless),? (?:there (?:are|is)|it) (?:still|remains?) (?:some|several|many|a number of) (?:challenges|limitations|problems|issues)",
    r"the proposed (?:method|approach|framework|model|algorithm) (?:outperforms|achieves|shows)",
    r"(?:compared|comparison) with (?:other|existing|traditional|state-of-the-art|previous) (?:methods|approaches)",
    r"the experimental results (?:demonstrate|show|indicate|verify|validate)",
    r"(?:first|firstly),? .{10,80}(?:second|secondly),? .{10,80}(?:third|thirdly)",
    r"(?:is|are) summarized in (?:table|fig)",
    r"as shown in (?:table|fig|figure)",
]

# Template abstract patterns (sentence role sequences)
ABSTRACT_SENTENCE_ROLES = {
    "background": re.compile(
        r"^(?:in recent|with the|the |over the past|currently|nowadays)", re.I
    ),
    "objective": re.compile(
        r"^(?:this study|in this|we (?:aim|propose|present|develop)|the (?:aim|purpose|objective))",
        re.I,
    ),
    "methods": re.compile(
        r"^(?:we (?:use|used|employ|collect|train|applied|develop)|a total of|data (?:were|was)|the dataset)",
        re.I,
    ),
    "results": re.compile(
        r"^(?:the results|our (?:results|method|model|approach)|the (?:proposed|accuracy|AUC|performance))",
        re.I,
    ),
    "conclusion": re.compile(
        r"^(?:(?:in )?conclusion|the (?:proposed|study)|our (?:findings|results) (?:suggest|indicate|demonstrate)|this (?:study|work))",
        re.I,
    ),
}


@dataclass
class StructureScore:
    """Structural analysis features for a single paper."""

    # Section structure
    has_standard_imrad: bool = False  # Exact IMRAD order
    section_count: int = 0
    standard_section_fraction: float = 0.0  # fraction of sections with standard names

    # Boilerplate
    boilerplate_sentence_count: int = 0
    boilerplate_density: float = 0.0  # per 1000 words

    # Abstract template score
    abstract_template_score: float = 0.0  # 0-1, how closely it follows template pattern

    # Text uniformity
    sentence_length_cv: float = 0.0  # coefficient of variation of sentence lengths
    paragraph_length_cv: float = 0.0
    vocabulary_diversity: float = 0.0  # type-token ratio

    # Methods section
    methods_length_ratio: float = 0.0  # methods length / total length

    def to_dict(self) -> dict:
        return {
            "has_standard_imrad": self.has_standard_imrad,
            "section_count": self.section_count,
            "standard_section_fraction": round(self.standard_section_fraction, 4),
            "boilerplate_sentence_count": self.boilerplate_sentence_count,
            "boilerplate_density": round(self.boilerplate_density, 4),
            "abstract_template_score": round(self.abstract_template_score, 4),
            "sentence_length_cv": round(self.sentence_length_cv, 4),
            "paragraph_length_cv": round(self.paragraph_length_cv, 4),
            "vocabulary_diversity": round(self.vocabulary_diversity, 4),
            "methods_length_ratio": round(self.methods_length_ratio, 4),
        }


def _normalise_heading(heading: str) -> str:
    """Map a section heading to its canonical name."""
    h = heading.strip().lower()
    h = re.sub(r"^\d+[\.\)]\s*", "", h)  # Remove numbering
    if h in SECTION_ALIASES:
        return SECTION_ALIASES[h]
    # Check if any standard section name is contained
    for std in STANDARD_SECTIONS:
        if std in h:
            return std
    return h


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple rule-based)."""
    # Split on period/question/exclamation followed by space and uppercase
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def score_section_structure(section_headings: list[str]) -> dict:
    """Analyse section heading structure.

    Returns features about how closely the paper follows standard
    IMRAD (Introduction/Methods/Results/Discussion) format.
    """
    if not section_headings:
        return {
            "has_standard_imrad": False,
            "section_count": 0,
            "standard_section_fraction": 0.0,
        }

    normalised = [_normalise_heading(h) for h in section_headings]
    section_count = len(normalised)

    # Check how many sections map to standard names
    standard_count = sum(1 for h in normalised if h in STANDARD_SECTIONS)
    standard_fraction = standard_count / section_count if section_count else 0.0

    # Check for exact IMRAD order
    standard_order = [h for h in normalised if h in STANDARD_SECTIONS]
    expected_order = [s for s in STANDARD_SECTIONS if s in standard_order]
    has_imrad = standard_order == expected_order and len(standard_order) >= 4

    return {
        "has_standard_imrad": has_imrad,
        "section_count": section_count,
        "standard_section_fraction": standard_fraction,
    }


def score_boilerplate(text: str) -> dict:
    """Count boilerplate/formulaic sentences in text."""
    if not text:
        return {"boilerplate_sentence_count": 0, "boilerplate_density": 0.0}

    compiled_patterns = [
        re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS
    ]

    matches = 0
    for pattern in compiled_patterns:
        matches += len(pattern.findall(text))

    word_count = len(text.split())
    density = (matches / word_count * 1000) if word_count > 0 else 0.0

    return {
        "boilerplate_sentence_count": matches,
        "boilerplate_density": density,
    }


def score_abstract_template(abstract: str) -> float:
    """Score how closely an abstract follows a formulaic template pattern.

    Paper mill abstracts typically follow: background -> objective -> methods
    -> results -> conclusion, with very predictable sentence openings.

    Returns:
        Score from 0 (not template-like) to 1 (highly formulaic)
    """
    if not abstract:
        return 0.0

    sentences = _split_sentences(abstract)
    if len(sentences) < 3:
        return 0.0

    # Classify each sentence's role
    roles = []
    for sent in sentences:
        sent_clean = sent.strip()
        matched_role = None
        for role, pattern in ABSTRACT_SENTENCE_ROLES.items():
            if pattern.match(sent_clean):
                matched_role = role
                break
        roles.append(matched_role)

    # Score: what fraction of sentences match a role?
    matched_count = sum(1 for r in roles if r is not None)
    match_fraction = matched_count / len(sentences)

    # Bonus: do the matched roles appear in expected order?
    expected_order = ["background", "objective", "methods", "results", "conclusion"]
    matched_roles = [r for r in roles if r is not None]

    order_score = 0.0
    if len(matched_roles) >= 3:
        # Check if the matched roles appear in the expected order
        expected_positions = {role: i for i, role in enumerate(expected_order)}
        positions = [expected_positions.get(r, -1) for r in matched_roles if r in expected_positions]
        if positions:
            # Count adjacent pairs in correct order
            correct_pairs = sum(1 for i in range(len(positions) - 1) if positions[i] <= positions[i + 1])
            order_score = correct_pairs / max(len(positions) - 1, 1)

    # Combined score
    return 0.6 * match_fraction + 0.4 * order_score


def score_text_uniformity(text: str) -> dict:
    """Measure text uniformity features.

    Paper mill text tends to have more uniform sentence lengths and
    lower vocabulary diversity than human-written text.
    """
    if not text or len(text) < 100:
        return {
            "sentence_length_cv": 0.0,
            "paragraph_length_cv": 0.0,
            "vocabulary_diversity": 0.0,
        }

    # Sentence length variation
    sentences = _split_sentences(text)
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        sentence_cv = std_len / mean_len if mean_len > 0 else 0.0
    else:
        sentence_cv = 0.0

    # Paragraph length variation
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    if len(paragraphs) >= 2:
        p_lengths = [len(p.split()) for p in paragraphs]
        p_mean = np.mean(p_lengths)
        p_std = np.std(p_lengths)
        paragraph_cv = p_std / p_mean if p_mean > 0 else 0.0
    else:
        paragraph_cv = 0.0

    # Vocabulary diversity (type-token ratio on first 1000 tokens)
    tokens = text.lower().split()[:1000]
    if tokens:
        vocabulary_diversity = len(set(tokens)) / len(tokens)
    else:
        vocabulary_diversity = 0.0

    return {
        "sentence_length_cv": sentence_cv,
        "paragraph_length_cv": paragraph_cv,
        "vocabulary_diversity": vocabulary_diversity,
    }


def score_paper(
    abstract: str = "",
    body_text: str = "",
    section_headings: Optional[list[str]] = None,
    sections: Optional[list[dict]] = None,
) -> StructureScore:
    """Compute all structure scores for a paper.

    Args:
        abstract: Paper abstract text
        body_text: Full body text
        section_headings: List of section heading strings
        sections: List of {"heading": str, "text": str} dicts

    Returns:
        StructureScore with all computed features
    """
    result = StructureScore()

    # Section structure
    if section_headings:
        structure = score_section_structure(section_headings)
        result.has_standard_imrad = structure["has_standard_imrad"]
        result.section_count = structure["section_count"]
        result.standard_section_fraction = structure["standard_section_fraction"]

    # Boilerplate in body text
    full_text = f"{abstract} {body_text}".strip()
    if full_text:
        bp = score_boilerplate(full_text)
        result.boilerplate_sentence_count = bp["boilerplate_sentence_count"]
        result.boilerplate_density = bp["boilerplate_density"]

    # Abstract template score
    result.abstract_template_score = score_abstract_template(abstract)

    # Text uniformity
    if body_text:
        uniformity = score_text_uniformity(body_text)
        result.sentence_length_cv = uniformity["sentence_length_cv"]
        result.paragraph_length_cv = uniformity["paragraph_length_cv"]
        result.vocabulary_diversity = uniformity["vocabulary_diversity"]

    # Methods length ratio
    if sections:
        total_len = sum(len(s.get("text", "")) for s in sections)
        methods_len = sum(
            len(s.get("text", ""))
            for s in sections
            if _normalise_heading(s.get("heading", "")) == "methods"
        )
        result.methods_length_ratio = methods_len / total_len if total_len > 0 else 0.0

    return result


if __name__ == "__main__":
    # Test with a formulaic abstract
    test_abstract = """
    In recent years, deep learning has attracted widespread attention in medical
    image analysis. This study proposes a novel convolutional neural network for
    breast cancer detection using mammography images. We used a dataset of 5000
    mammography images collected from three hospitals. Data were divided into
    training (80%) and testing (20%) sets. The results show that our proposed
    method achieved an accuracy of 95.6% and an AUC of 0.978, outperforming
    existing methods. In conclusion, the proposed method demonstrates promising
    performance for breast cancer detection.
    """

    test_headings = [
        "1. Introduction",
        "2. Materials and Methods",
        "2.1 Dataset",
        "2.2 Proposed Method",
        "2.3 Statistical Analysis",
        "3. Results",
        "4. Discussion",
        "5. Conclusion",
    ]

    score = score_paper(
        abstract=test_abstract,
        section_headings=test_headings,
    )
    print("Structure Score:")
    for k, v in score.to_dict().items():
        print(f"  {k}: {v}")
