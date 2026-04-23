"""
Bibliometric anomaly features from OpenAlex metadata.

These features capture suspicious author and publication patterns
without requiring full text — just metadata from OpenAlex:
- Extremely prolific authors (>20 papers/year)
- Low h-index relative to publication count
- Journal quality indicators
- Unusually fast acceptance times
- Geographic patterns
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Thresholds based on bibliometric literature
PROLIFIC_THRESHOLD = 20  # papers/year considered unusually high
LOW_H_INDEX_RATIO = 0.1  # h-index / total_papers ratio considered low


@dataclass
class BibliometricFeatures:
    """Bibliometric anomaly features for a single paper."""

    author_count: int = 0
    first_author_paper_count: int = 0
    first_author_citation_count: int = 0
    first_author_h_index: float = 0.0
    first_author_papers_per_year: float = 0.0
    corresponding_country: str = ""
    institution_count: int = 0  # unique institutions on paper
    country_count: int = 0  # unique countries on paper
    journal_h_index: float = 0.0
    journal_paper_count: int = 0
    is_oa: bool = False
    oa_status: str = ""
    cited_by_count: int = 0

    # Derived flags
    is_prolific_author: bool = False
    low_h_index_ratio: bool = False

    def to_dict(self) -> dict:
        return {
            "author_count": self.author_count,
            "first_author_paper_count": self.first_author_paper_count,
            "first_author_citation_count": self.first_author_citation_count,
            "first_author_h_index": self.first_author_h_index,
            "first_author_papers_per_year": round(self.first_author_papers_per_year, 2),
            "corresponding_country": self.corresponding_country,
            "institution_count": self.institution_count,
            "country_count": self.country_count,
            "journal_h_index": self.journal_h_index,
            "journal_paper_count": self.journal_paper_count,
            "is_oa": self.is_oa,
            "oa_status": self.oa_status,
            "cited_by_count": self.cited_by_count,
            "is_prolific_author": self.is_prolific_author,
            "low_h_index_ratio": self.low_h_index_ratio,
        }


def compute_from_openalex_record(record: dict) -> BibliometricFeatures:
    """Extract bibliometric features from a corpus record.

    Args:
        record: A single paper record dict (as produced by openalex_collector)

    Returns:
        BibliometricFeatures dataclass
    """
    features = BibliometricFeatures()

    features.author_count = record.get("author_count", 0)
    features.cited_by_count = record.get("cited_by_count", 0)
    features.is_oa = record.get("is_oa", False)
    features.oa_status = record.get("oa_status", "")

    # Country info
    countries = str(record.get("all_countries", "") or "")
    if countries and countries != "nan":
        country_list = [c.strip() for c in countries.split(";") if c.strip()]
        features.country_count = len(country_list)
    features.corresponding_country = str(record.get("corresponding_countries", "") or "")

    return features


def enrich_author_features(
    features: BibliometricFeatures,
    author_works_count: int = 0,
    author_cited_by_count: int = 0,
    author_h_index: float = 0.0,
    author_first_publication_year: int = 0,
    current_year: int = 2025,
) -> BibliometricFeatures:
    """Enrich features with author-level data from OpenAlex.

    This requires a separate API call to get author details,
    so it's kept separate from the basic record extraction.
    """
    features.first_author_paper_count = author_works_count
    features.first_author_citation_count = author_cited_by_count
    features.first_author_h_index = author_h_index

    # Papers per year
    if author_first_publication_year and author_first_publication_year < current_year:
        career_years = current_year - author_first_publication_year
        features.first_author_papers_per_year = (
            author_works_count / max(career_years, 1)
        )
    features.is_prolific_author = (
        features.first_author_papers_per_year > PROLIFIC_THRESHOLD
    )

    # H-index ratio
    if author_works_count > 10:
        ratio = author_h_index / author_works_count
        features.low_h_index_ratio = ratio < LOW_H_INDEX_RATIO

    return features
