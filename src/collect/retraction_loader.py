"""
Load and filter Retraction Watch data for medical AI paper retractions.

The Retraction Watch database provides retraction records that serve as
positive labels for our classifier — papers retracted for fabrication,
fraud, or paper mill origin in the medical AI domain.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Retraction reasons that suggest paper mill or fabrication origin
FRAUD_REASONS = [
    "Concerns/Issues About Data",
    "Fabrication/Falsification of Data",
    "Fabrication/Falsification of Images",
    "Fake Peer Review",
    "Manipulation of Images",
    "Manipulation of Results",
    "Paper Mill",
    "Concerns/Issues About Image Integrity",
    "Unreliable Data",
    "Concerns/Issues About Referencing/Attributions",
    "Duplication of Article",
    "Plagiarism of Data",
]

# Keywords to identify medical AI papers in retraction records
MEDICAL_AI_KEYWORDS = [
    r"\bdeep learning\b",
    r"\bmachine learning\b",
    r"\bartificial intelligence\b",
    r"\bneural network\b",
    r"\bconvolutional\b",
    r"\brandom forest\b",
    r"\bsupport vector\b",
    r"\bimage classification\b",
    r"\bimage segmentation\b",
    r"\bnatural language processing\b",
    r"\bprediction model\b",
    r"\bpredictive model\b",
    r"\bdiagnostic model\b",
    r"\bprognostic model\b",
    r"\bautomated detection\b",
    r"\bcomputer.aided\b",
    r"\bfeature extraction\b",
    r"\bXGBoost\b",
    r"\bLSTM\b",
    r"\btransformer\b",
    r"\bGAN\b",
    r"\bU-?Net\b",
    r"\bResNet\b",
    r"\bVGG\b",
    r"\bBERT\b",
]


def load_retraction_watch(
    filepath: Path,
    filter_medical_ai: bool = True,
    filter_fraud: bool = True,
    year_range: Optional[tuple[int, int]] = None,
) -> pd.DataFrame:
    """Load Retraction Watch data and filter to medical AI retractions.

    The Retraction Watch database can be requested for research use
    from retractiondatabase.org. Expected format: CSV/Excel with columns
    including Title, Journal, Subject, Reason, DOI, PMID, Date.

    Args:
        filepath: Path to the Retraction Watch data file
        filter_medical_ai: If True, filter to papers matching AI/ML keywords
        filter_fraud: If True, filter to fabrication/fraud-related reasons
        year_range: Optional (start, end) year filter

    Returns:
        Filtered DataFrame of retraction records
    """
    filepath = Path(filepath)

    # Support CSV and Excel
    if filepath.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    logger.info(f"Loaded {len(df)} retraction records from {filepath.name}")

    # Standardise column names (Retraction Watch uses varying formats)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "title" in col_lower and "article" in col_lower:
            col_map[col] = "title"
        elif col_lower == "title":
            col_map[col] = "title"
        elif "journal" in col_lower:
            col_map[col] = "journal"
        elif "subject" in col_lower:
            col_map[col] = "subject"
        elif "reason" in col_lower:
            col_map[col] = "reason"
        elif col_lower == "doi":
            col_map[col] = "doi"
        elif "pmid" in col_lower:
            col_map[col] = "pmid"
        elif "date" in col_lower and "retract" in col_lower:
            col_map[col] = "retraction_date"
        elif "original" in col_lower and "date" in col_lower:
            col_map[col] = "original_date"

    df = df.rename(columns=col_map)

    # Parse dates and extract year
    if "retraction_date" in df.columns:
        df["retraction_date"] = pd.to_datetime(df["retraction_date"], errors="coerce")
        df["retraction_year"] = df["retraction_date"].dt.year

    if "original_date" in df.columns:
        df["original_date"] = pd.to_datetime(df["original_date"], errors="coerce")
        df["publication_year"] = df["original_date"].dt.year

    # Filter by year if specified
    if year_range:
        start, end = year_range
        year_col = "publication_year" if "publication_year" in df.columns else "retraction_year"
        if year_col in df.columns:
            df = df[df[year_col].between(start, end)]
            logger.info(f"After year filter ({start}-{end}): {len(df)} records")

    # Filter to fraud/fabrication reasons
    if filter_fraud and "reason" in df.columns:
        reason_pattern = "|".join(re.escape(r) for r in FRAUD_REASONS)
        fraud_mask = df["reason"].str.contains(reason_pattern, case=False, na=False)
        df = df[fraud_mask]
        logger.info(f"After fraud reason filter: {len(df)} records")

    # Filter to medical AI papers
    if filter_medical_ai:
        # Check title and subject fields for AI/ML keywords
        search_cols = ["title", "subject"]
        available_cols = [c for c in search_cols if c in df.columns]

        if available_cols:
            combined_text = df[available_cols].fillna("").agg(" ".join, axis=1)
            ai_pattern = "|".join(MEDICAL_AI_KEYWORDS)
            ai_mask = combined_text.str.contains(ai_pattern, case=False, na=False)
            df = df[ai_mask]
            logger.info(f"After medical AI filter: {len(df)} records")

    return df.reset_index(drop=True)


def get_retracted_dois(df: pd.DataFrame) -> set[str]:
    """Extract set of DOIs from retraction records for matching."""
    if "doi" not in df.columns:
        return set()
    return set(df["doi"].dropna().str.strip().str.lower())


def get_retracted_pmids(df: pd.DataFrame) -> set[str]:
    """Extract set of PMIDs from retraction records for matching."""
    if "pmid" not in df.columns:
        return set()
    return set(df["pmid"].dropna().astype(str).str.strip())


def match_with_corpus(
    retraction_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match retraction records against the collected paper corpus.

    Matches on DOI (primary) and PMID (secondary).

    Returns:
        Corpus DataFrame with added 'is_retracted' and 'retraction_reason' columns
    """
    retracted_dois = get_retracted_dois(retraction_df)
    retracted_pmids = get_retracted_pmids(retraction_df)

    # Normalise DOIs in corpus
    corpus_df = corpus_df.copy()
    if "doi" in corpus_df.columns:
        corpus_doi_norm = corpus_df["doi"].fillna("").str.strip().str.lower()
    else:
        corpus_doi_norm = pd.Series("", index=corpus_df.index)

    if "pmid" in corpus_df.columns:
        corpus_pmid_norm = corpus_df["pmid"].fillna("").astype(str).str.strip()
    else:
        corpus_pmid_norm = pd.Series("", index=corpus_df.index)

    # Match
    doi_match = corpus_doi_norm.isin(retracted_dois)
    pmid_match = corpus_pmid_norm.isin(retracted_pmids)
    corpus_df["is_retracted"] = doi_match | pmid_match

    # Add retraction reason where matched
    doi_to_reason = {}
    if "doi" in retraction_df.columns and "reason" in retraction_df.columns:
        for _, row in retraction_df.iterrows():
            doi = str(row.get("doi", "")).strip().lower()
            if doi:
                doi_to_reason[doi] = row.get("reason", "")

    corpus_df["retraction_reason"] = corpus_doi_norm.map(doi_to_reason).fillna("")

    n_matched = corpus_df["is_retracted"].sum()
    logger.info(
        f"Matched {n_matched} retracted papers in corpus of {len(corpus_df)}"
    )

    return corpus_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(__file__).resolve().parents[2] / "data" / "retraction_watch"
    rw_files = list(data_dir.glob("*retract*"))

    if rw_files:
        df = load_retraction_watch(rw_files[0])
        print(f"\nFiltered retractions: {len(df)}")
        if len(df) > 0:
            print(f"DOIs available: {df['doi'].notna().sum()}")
            print(f"\nSample titles:")
            for t in df["title"].head(5):
                print(f"  - {t[:80]}")
    else:
        print(f"No Retraction Watch data files found in {data_dir}")
        print("Please download from retractiondatabase.org and place in data/retraction_watch/")
