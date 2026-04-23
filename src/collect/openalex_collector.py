"""
Collect medical AI paper metadata and abstracts from OpenAlex.

Uses pyalex to query the OpenAlex API for papers at the intersection of
medicine and AI/ML, retrieving metadata needed for paper mill detection.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pyalex import Works
import pyalex

logger = logging.getLogger(__name__)

# Set a polite email for faster API access (OpenAlex "polite pool")
# Users should set this to their own email
pyalex.config.email = "hayden.farquhar@example.com"

# OpenAlex concept IDs for filtering
# These are stable IDs from the OpenAlex concepts taxonomy
CONCEPT_IDS = {
    # AI/ML concepts
    "artificial_intelligence": "C154945302",
    "machine_learning": "C119857082",
    "deep_learning": "C108827166",
    "neural_network": "C50644808",
    "natural_language_processing": "C204321447",
    "computer_vision": "C31972630",
    # Medical concepts
    "medicine": "C71924100",
    "clinical_medicine": "C118552586",
    "radiology": "C153294291",
    "pathology": "C126838900",
    "oncology": "C126322002",
    "surgery": "C159985019",
    "cardiology": "C49204034",
}

# Search terms for medical AI papers (used in text search)
MEDICAL_AI_SEARCH_TERMS = [
    "deep learning AND (diagnosis OR prognosis OR medical imaging)",
    "machine learning AND (clinical OR patient OR disease)",
    "artificial intelligence AND (radiology OR pathology OR oncology)",
    "neural network AND (medical image OR clinical prediction)",
    "convolutional neural network AND (diagnosis OR detection OR segmentation)",
]


def reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Reconstruct abstract text from OpenAlex inverted index format.

    OpenAlex stores abstracts as {word: [position, ...]} mappings.
    We reconstruct the original text by placing each word at its positions.
    """
    if not inverted_index:
        return None

    # Build position -> word mapping
    word_positions = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions[pos] = word

    if not word_positions:
        return None

    # Reconstruct text in order
    max_pos = max(word_positions.keys())
    words = [word_positions.get(i, "") for i in range(max_pos + 1)]
    return " ".join(w for w in words if w)


def extract_paper_record(work: dict) -> dict:
    """Extract relevant fields from an OpenAlex work record."""

    # Get authorships info
    authorships = work.get("authorships", [])
    first_author = authorships[0] if authorships else {}
    first_author_name = first_author.get("author", {}).get("display_name", "")
    first_author_id = first_author.get("author", {}).get("id", "")

    # Get corresponding author (last author with is_corresponding=True, or last author)
    corresponding = next(
        (a for a in authorships if a.get("is_corresponding")),
        authorships[-1] if authorships else {}
    )
    corresponding_countries = [
        inst.get("country_code", "")
        for inst in corresponding.get("institutions", [])
        if inst.get("country_code")
    ]

    # Get all author institution countries
    all_countries = set()
    for authorship in authorships:
        for inst in authorship.get("institutions", []):
            cc = inst.get("country_code")
            if cc:
                all_countries.add(cc)

    # Source (journal) info
    source = work.get("primary_location", {}).get("source", {}) or {}

    # Reconstruct abstract
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    # Get IDs
    ids = work.get("ids", {})

    return {
        "openalex_id": work.get("id", ""),
        "doi": work.get("doi", ""),
        "pmid": ids.get("pmid", ""),
        "pmcid": ids.get("pmcid", ""),
        "title": work.get("title", ""),
        "abstract": abstract,
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date", ""),
        "type": work.get("type", ""),
        "cited_by_count": work.get("cited_by_count", 0),
        # Journal info
        "journal_name": source.get("display_name", ""),
        "journal_issn": source.get("issn_l", ""),
        "journal_type": source.get("type", ""),
        # Author info
        "author_count": len(authorships),
        "first_author_name": first_author_name,
        "first_author_id": first_author_id,
        # All author IDs (semicolon-separated, for co-authorship network analysis)
        "author_ids": ";".join(
            a.get("author", {}).get("id", "")
            for a in authorships
            if a.get("author", {}).get("id")
        ),
        "corresponding_countries": ";".join(corresponding_countries),
        "all_countries": ";".join(sorted(all_countries)),
        # Concepts/topics
        "concepts": ";".join(
            c.get("display_name", "")
            for c in work.get("concepts", [])[:10]
        ),
        # Open access status
        "is_oa": work.get("open_access", {}).get("is_oa", False),
        "oa_status": work.get("open_access", {}).get("oa_status", ""),
        # Referenced works (for citation analysis)
        "referenced_works_count": len(work.get("referenced_works", [])),
        "referenced_works": ";".join(work.get("referenced_works", [])[:50]),
    }


def collect_medical_ai_papers(
    year_range: tuple[int, int] = (2018, 2025),
    max_papers: int = 1000,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Collect medical AI papers from OpenAlex.

    Searches for papers at the intersection of AI/ML and medicine,
    published in the specified year range.

    Args:
        year_range: (start_year, end_year) inclusive
        max_papers: Maximum number of papers to retrieve
        output_dir: Directory to save results (as CSV and JSON)

    Returns:
        DataFrame with paper metadata and abstracts
    """
    start_year, end_year = year_range
    logger.info(
        f"Collecting up to {max_papers} medical AI papers ({start_year}-{end_year})"
    )

    all_records = []
    seen_ids = set()

    # Strategy: Use OpenAlex search with concept filtering
    # We combine AI concepts with medical concepts for targeted retrieval
    ai_concept_ids = [
        CONCEPT_IDS["artificial_intelligence"],
        CONCEPT_IDS["machine_learning"],
        CONCEPT_IDS["deep_learning"],
    ]
    medical_concept_ids = [
        CONCEPT_IDS["medicine"],
        CONCEPT_IDS["clinical_medicine"],
    ]

    # Build filter: papers with BOTH an AI concept AND a medical concept
    for ai_concept in ai_concept_ids:
        if len(all_records) >= max_papers:
            break

        for med_concept in medical_concept_ids:
            if len(all_records) >= max_papers:
                break

            remaining = max_papers - len(all_records)
            logger.info(
                f"Querying AI concept {ai_concept} + medical concept {med_concept} "
                f"({len(all_records)}/{max_papers} collected)"
            )

            try:
                query = (
                    Works()
                    .filter(
                        concepts={"id": ai_concept},
                        publication_year=f"{start_year}-{end_year}",
                        type="article",
                    )
                    .filter(concepts={"id": med_concept})
                    .sort(cited_by_count="desc")
                )

                count = 0
                for page in query.paginate(per_page=200):
                    for work in page:
                        oa_id = work.get("id", "")
                        if oa_id in seen_ids:
                            continue
                        seen_ids.add(oa_id)

                        record = extract_paper_record(work)
                        # Only keep papers with abstracts (needed for NLP analysis)
                        if record["abstract"]:
                            all_records.append(record)
                            count += 1

                        if len(all_records) >= max_papers:
                            break

                    if len(all_records) >= max_papers:
                        break

                logger.info(f"  -> Retrieved {count} new papers from this query")

            except Exception as e:
                logger.warning(f"Error querying OpenAlex: {e}")
                continue

    logger.info(f"Total papers collected: {len(all_records)}")

    df = pd.DataFrame(all_records)

    # Save if output directory provided
    if output_dir and len(df) > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "medical_ai_papers.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} papers to {csv_path}")

        # Also save as JSON for full fidelity
        json_path = output_dir / "medical_ai_papers.json"
        df.to_json(json_path, orient="records", indent=2)
        logger.info(f"Saved JSON to {json_path}")

    return df


def collect_papers_by_search(
    search_queries: Optional[list[str]] = None,
    year_range: tuple[int, int] = (2018, 2025),
    max_per_query: int = 200,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Alternative collection using text search queries.

    Useful for supplementing concept-based collection with specific
    search terms targeting medical AI subfields.
    """
    if search_queries is None:
        search_queries = MEDICAL_AI_SEARCH_TERMS

    start_year, end_year = year_range
    all_records = []
    seen_ids = set()

    for query_text in search_queries:
        logger.info(f"Searching: {query_text}")

        try:
            query = (
                Works()
                .search(query_text)
                .filter(
                    publication_year=f"{start_year}-{end_year}",
                    type="article",
                )
                .sort(cited_by_count="desc")
            )

            count = 0
            for page in query.paginate(per_page=200):
                for work in page:
                    oa_id = work.get("id", "")
                    if oa_id in seen_ids:
                        continue
                    seen_ids.add(oa_id)

                    record = extract_paper_record(work)
                    if record["abstract"]:
                        all_records.append(record)
                        count += 1

                    if count >= max_per_query:
                        break

                if count >= max_per_query:
                    break

            logger.info(f"  -> {count} papers")

        except Exception as e:
            logger.warning(f"Error searching OpenAlex: {e}")
            continue

    df = pd.DataFrame(all_records)

    if output_dir and len(df) > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "medical_ai_papers_search.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} papers to {csv_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    output = Path(__file__).resolve().parents[2] / "data" / "openalex"
    df = collect_medical_ai_papers(
        year_range=(2018, 2025),
        max_papers=1000,
        output_dir=output,
    )
    print(f"\nCollected {len(df)} papers")
    print(f"Year range: {df['publication_year'].min()}-{df['publication_year'].max()}")
    print(f"Journals: {df['journal_name'].nunique()} unique")
    print(f"With PMC ID: {df['pmcid'].notna().sum()}")
