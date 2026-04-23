"""
Analyse citation patterns for anomalies indicative of paper mills.

Paper mills exhibit distinctive citation patterns:
- Citation rings (small groups of papers citing each other excessively)
- Irrelevant citations (padding reference lists with unrelated papers)
- Extreme recency bias (only citing very recent papers)
- High self-citation rates
- Unusual reference counts (too few or too many for the field)
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CitationFeatures:
    """Citation-based features for a single paper."""

    reference_count: int = 0
    self_citation_rate: float = 0.0  # fraction of refs by same authors
    citation_recency_bias: float = 0.0  # fraction of refs from last 2 years
    median_citation_age: float = 0.0  # median age of cited papers in years
    citation_concentration: float = 0.0  # HHI of cited journals
    unique_cited_journals: int = 0
    max_single_journal_share: float = 0.0  # most-cited journal's share

    def to_dict(self) -> dict:
        return {
            "reference_count": self.reference_count,
            "self_citation_rate": round(self.self_citation_rate, 4),
            "citation_recency_bias": round(self.citation_recency_bias, 4),
            "median_citation_age": round(self.median_citation_age, 2),
            "citation_concentration": round(self.citation_concentration, 4),
            "unique_cited_journals": self.unique_cited_journals,
            "max_single_journal_share": round(self.max_single_journal_share, 4),
        }


def compute_self_citation_rate(
    paper_author_ids: list[str],
    referenced_author_ids: list[list[str]],
) -> float:
    """Compute fraction of references that share authors with the paper.

    Args:
        paper_author_ids: OpenAlex author IDs for the paper being analysed
        referenced_author_ids: List of author ID lists, one per referenced work

    Returns:
        Fraction of references with overlapping authors
    """
    if not referenced_author_ids or not paper_author_ids:
        return 0.0

    paper_set = set(paper_author_ids)
    self_cited = sum(
        1 for ref_authors in referenced_author_ids
        if paper_set & set(ref_authors)
    )
    return self_cited / len(referenced_author_ids)


def compute_citation_recency(
    paper_year: int,
    referenced_years: list[int],
    recency_window: int = 2,
) -> dict:
    """Compute citation recency features.

    Args:
        paper_year: Publication year of the paper
        referenced_years: Publication years of referenced works
        recency_window: Years to consider as "recent"

    Returns:
        Dict with recency_bias and median_age
    """
    if not referenced_years:
        return {"citation_recency_bias": 0.0, "median_citation_age": 0.0}

    ages = [paper_year - y for y in referenced_years if y and y <= paper_year]
    if not ages:
        return {"citation_recency_bias": 0.0, "median_citation_age": 0.0}

    recent_count = sum(1 for a in ages if a <= recency_window)
    recency_bias = recent_count / len(ages)
    median_age = float(np.median(ages))

    return {
        "citation_recency_bias": recency_bias,
        "median_citation_age": median_age,
    }


def compute_citation_concentration(referenced_journals: list[str]) -> dict:
    """Compute citation concentration across journals.

    High concentration (citing many papers from one journal) can indicate
    citation manipulation or ring behaviour.

    Uses Herfindahl-Hirschman Index (HHI) of journal shares.
    """
    if not referenced_journals:
        return {
            "citation_concentration": 0.0,
            "unique_cited_journals": 0,
            "max_single_journal_share": 0.0,
        }

    # Filter out empty/unknown
    journals = [j for j in referenced_journals if j]
    if not journals:
        return {
            "citation_concentration": 0.0,
            "unique_cited_journals": 0,
            "max_single_journal_share": 0.0,
        }

    counts = Counter(journals)
    total = sum(counts.values())
    shares = [c / total for c in counts.values()]

    hhi = sum(s ** 2 for s in shares)
    max_share = max(shares)

    return {
        "citation_concentration": hhi,
        "unique_cited_journals": len(counts),
        "max_single_journal_share": max_share,
    }


def detect_citation_ring(
    corpus_df: pd.DataFrame,
    min_ring_size: int = 3,
    min_mutual_citations: int = 2,
) -> pd.DataFrame:
    """Detect potential citation rings in the corpus.

    A citation ring is a small group of papers that cite each other
    disproportionately compared to other papers.

    This requires referenced_works columns with OpenAlex IDs.

    Args:
        corpus_df: DataFrame with 'openalex_id' and 'referenced_works' columns
        min_ring_size: Minimum number of papers to form a ring
        min_mutual_citations: Minimum mutual citations to flag

    Returns:
        DataFrame with citation ring scores added
    """
    if "referenced_works" not in corpus_df.columns:
        corpus_df["citation_ring_score"] = 0.0
        return corpus_df

    # Build citation graph within the corpus
    corpus_ids = set(corpus_df["openalex_id"].dropna())

    # For each paper, find how many corpus papers it cites
    internal_citation_counts = {}
    for _, row in corpus_df.iterrows():
        oa_id = row.get("openalex_id", "")
        refs = str(row.get("referenced_works", "")).split(";")
        internal_refs = [r.strip() for r in refs if r.strip() in corpus_ids]
        internal_citation_counts[oa_id] = internal_refs

    # For each paper, count how many corpus papers cite it
    cited_by = {oa_id: [] for oa_id in corpus_ids}
    for citer, refs in internal_citation_counts.items():
        for ref in refs:
            if ref in cited_by:
                cited_by[ref].append(citer)

    # Compute ring score: for each paper, what fraction of its internal
    # citations are reciprocal (the cited paper also cites back)?
    ring_scores = {}
    for oa_id in corpus_ids:
        my_refs = set(internal_citation_counts.get(oa_id, []))
        my_citers = set(cited_by.get(oa_id, []))

        if not my_refs:
            ring_scores[oa_id] = 0.0
            continue

        mutual = my_refs & my_citers
        ring_scores[oa_id] = len(mutual) / len(my_refs) if my_refs else 0.0

    corpus_df = corpus_df.copy()
    corpus_df["citation_ring_score"] = corpus_df["openalex_id"].map(ring_scores).fillna(0.0)

    n_flagged = (corpus_df["citation_ring_score"] > 0).sum()
    logger.info(f"Citation ring analysis: {n_flagged} papers with mutual citations")

    return corpus_df


def analyse_paper_citations(
    paper_year: int,
    reference_count: int,
    referenced_years: Optional[list[int]] = None,
    referenced_journals: Optional[list[str]] = None,
    paper_author_ids: Optional[list[str]] = None,
    referenced_author_ids: Optional[list[list[str]]] = None,
) -> CitationFeatures:
    """Compute all citation features for a single paper.

    This is a convenience function that combines all citation analyses.
    Not all inputs are required — missing data produces zero scores.
    """
    features = CitationFeatures(reference_count=reference_count)

    if referenced_years:
        recency = compute_citation_recency(paper_year, referenced_years)
        features.citation_recency_bias = recency["citation_recency_bias"]
        features.median_citation_age = recency["median_citation_age"]

    if referenced_journals:
        concentration = compute_citation_concentration(referenced_journals)
        features.citation_concentration = concentration["citation_concentration"]
        features.unique_cited_journals = concentration["unique_cited_journals"]
        features.max_single_journal_share = concentration["max_single_journal_share"]

    if paper_author_ids and referenced_author_ids:
        features.self_citation_rate = compute_self_citation_rate(
            paper_author_ids, referenced_author_ids
        )

    return features
