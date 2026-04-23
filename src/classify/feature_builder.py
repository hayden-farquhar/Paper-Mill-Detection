"""
Combine all detection features into a unified feature matrix.

Aggregates features from tortured phrases, structure scoring,
AI text detection, citation analysis, and bibliometric flags
into a single DataFrame suitable for classifier training.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.detect.tortured_phrases import TorturedPhraseDetector
from src.detect.structure_scorer import score_paper, score_abstract_template, score_boilerplate, score_text_uniformity
from src.detect.ai_text_detector import detect_ai_text
from src.detect.bibliometric_flags import compute_from_openalex_record
from src.detect.similarity_detector import add_similarity_features, add_methods_similarity_features
from src.detect.author_network import AuthorNetworkAnalyser
from src.analyse.geographic import prepare_geographic_features

logger = logging.getLogger(__name__)

# Feature columns produced by this module
FEATURE_COLUMNS = [
    # Tortured phrases
    "tortured_phrase_count",
    "tortured_phrase_density",
    "tortured_unique_count",
    # Structure
    "has_standard_imrad",
    "boilerplate_sentence_count",
    "boilerplate_density",
    "abstract_template_score",
    "sentence_length_cv",
    "vocabulary_diversity",
    # AI text
    "ai_vocabulary_diversity",
    "ai_hapax_ratio",
    "ai_sentence_uniformity",
    "ai_discourse_marker_density",
    "ai_llm_phrase_count",
    "ai_llm_phrase_density",
    "ai_avg_word_length",
    "ai_long_word_ratio",
    # Bibliometric
    "author_count",
    # NOTE: cited_by_count excluded — leaks retraction status (papers lose
    # citations post-retraction). Use only in downstream analysis, not as predictor.
    "country_count",
    "reference_count",
    # Citation (when available)
    "citation_recency_bias",
    "citation_concentration",
    # Cross-document similarity
    "sim_max",
    "sim_mean",
    "sim_n_high",
    "sim_n_moderate",
    # Author network
    "net_author_reuse_rate",
    "net_max_author_papers",
    "net_coauthor_density",
    "net_shared_coauthor_fraction",
    "net_clique_score",
    "net_pair_novelty",
    # Geographic (regional, not country-level)
    "geo_n_countries",
    "geo_is_international",
    # Methods-section similarity (full-text only, 0 when unavailable)
    "methods_sim_max",
    "methods_sim_mean",
    "methods_sim_n_high",
    "methods_in_cluster",
    # Interaction features
    "interaction_ai_x_tortured",
    "interaction_ai_x_template",
    "interaction_sim_x_boilerplate",
]


def build_features_from_abstracts(
    corpus_df: pd.DataFrame,
    tortured_detector: Optional[TorturedPhraseDetector] = None,
) -> pd.DataFrame:
    """Build feature matrix using only abstracts and metadata.

    This is the primary feature builder for the initial corpus where
    we only have OpenAlex metadata (abstracts, not full text).

    Args:
        corpus_df: DataFrame with columns including 'abstract', 'openalex_id',
                   and metadata fields from openalex_collector
        tortured_detector: Initialized detector (created if None)

    Returns:
        DataFrame with one row per paper and feature columns
    """
    if tortured_detector is None:
        tortured_detector = TorturedPhraseDetector()

    n = len(corpus_df)
    logger.info(f"Building features for {n} papers (abstract-only mode)")

    records = []

    for idx, row in corpus_df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing paper {idx}/{n}")

        abstract = row.get("abstract", "") or ""
        paper_id = row.get("openalex_id", str(idx))

        features = {"openalex_id": paper_id}

        # 1. Tortured phrase features
        tp_result = tortured_detector.detect(abstract)
        features.update({
            "tortured_phrase_count": tp_result.count,
            "tortured_phrase_density": tp_result.density,
            "tortured_unique_count": tp_result.unique_phrases,
        })

        # 2. Structure features (abstract-only)
        features["abstract_template_score"] = score_abstract_template(abstract)
        bp = score_boilerplate(abstract)
        features.update(bp)
        uniformity = score_text_uniformity(abstract)
        features.update(uniformity)
        # No section headings in abstract-only mode
        features["has_standard_imrad"] = False

        # 3. AI text features
        ai_features = detect_ai_text(abstract)
        features.update(ai_features.to_dict())

        # 4. Bibliometric features from metadata
        bib = compute_from_openalex_record(row.to_dict())
        features.update({
            "author_count": bib.author_count,
            "cited_by_count": bib.cited_by_count,
            "country_count": bib.country_count,
        })

        # 5. Basic citation features from metadata
        features["reference_count"] = row.get("referenced_works_count", 0)

        records.append(features)

    feature_df = pd.DataFrame(records)

    # 6. Cross-document similarity features (abstract-level)
    feature_df = add_similarity_features(feature_df, corpus_df)

    # 7. Author network features
    logger.info("Computing author network features...")
    network = AuthorNetworkAnalyser()
    network.build_network(corpus_df)
    net_df = network.compute_features_batch(corpus_df)
    feature_df = feature_df.merge(
        net_df, on="openalex_id", how="left", suffixes=("", "_net"),
    )
    # Fill missing network features with neutral values
    for col in ["net_author_reuse_rate", "net_coauthor_density",
                 "net_shared_coauthor_fraction", "net_clique_score"]:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(0.0)
    if "net_pair_novelty" in feature_df.columns:
        feature_df["net_pair_novelty"] = feature_df["net_pair_novelty"].fillna(1.0)
    if "net_max_author_papers" in feature_df.columns:
        feature_df["net_max_author_papers"] = feature_df["net_max_author_papers"].fillna(0)

    # 8. Geographic features (regional level, not country level)
    logger.info("Computing geographic features...")
    geo_df = prepare_geographic_features(corpus_df)
    feature_df = feature_df.merge(
        geo_df, on="openalex_id", how="left", suffixes=("", "_geo"),
    )
    for col in ["geo_n_countries", "geo_is_international"]:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(0.0)

    # 9. Interaction features — combinations more discriminative than
    # individual signals. AI text markers alone are ambiguous post-2023
    # (legitimate LLM-assisted writing), but combined with other mill signals
    # they become meaningful.
    feature_df["interaction_ai_x_tortured"] = (
        feature_df.get("ai_sentence_uniformity", 0)
        * feature_df.get("tortured_phrase_count", 0).clip(upper=5)
    )
    feature_df["interaction_ai_x_template"] = (
        feature_df.get("ai_sentence_uniformity", 0)
        * feature_df.get("abstract_template_score", 0)
    )
    feature_df["interaction_sim_x_boilerplate"] = (
        feature_df.get("sim_max", 0)
        * feature_df.get("boilerplate_density", 0)
    )

    # Initialise methods similarity columns to 0 (populated in full-text mode)
    for col in ["methods_sim_max", "methods_sim_mean",
                 "methods_sim_n_high", "methods_in_cluster"]:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    logger.info(f"Built feature matrix: {feature_df.shape}")

    return feature_df


def build_features_with_fulltext(
    corpus_df: pd.DataFrame,
    fulltext_data: list[dict],
    tortured_detector: Optional[TorturedPhraseDetector] = None,
) -> pd.DataFrame:
    """Build enriched feature matrix using full text from PMC.

    Args:
        corpus_df: DataFrame with paper metadata
        fulltext_data: List of parsed PMC articles (from pmc_fetcher.fetch_batch)
        tortured_detector: Initialized detector

    Returns:
        Feature DataFrame with both abstract and full-text features
    """
    if tortured_detector is None:
        tortured_detector = TorturedPhraseDetector()

    # Index full text by PMCID
    fulltext_index = {ft["pmcid"]: ft for ft in fulltext_data}

    # Start with abstract features
    feature_df = build_features_from_abstracts(corpus_df, tortured_detector)

    # Enrich with full-text features where available
    ft_features = []
    for _, row in corpus_df.iterrows():
        pmcid = row.get("pmcid", "")
        ft_data = fulltext_index.get(pmcid)

        if ft_data and ft_data.get("body_text"):
            body = ft_data["body_text"]
            headings = ft_data.get("section_headings", [])
            sections = ft_data.get("sections", [])

            # Full-text tortured phrases
            tp_full = tortured_detector.detect(body)

            # Full structure scoring
            structure = score_paper(
                abstract=row.get("abstract", ""),
                body_text=body,
                section_headings=headings,
                sections=sections,
            )

            # Full-text AI detection
            ai_full = detect_ai_text(body)

            ft_features.append({
                "openalex_id": row.get("openalex_id"),
                "has_fulltext": True,
                "ft_tortured_count": tp_full.count,
                "ft_tortured_density": tp_full.density,
                "ft_has_standard_imrad": structure.has_standard_imrad,
                "ft_section_count": structure.section_count,
                "ft_methods_length_ratio": structure.methods_length_ratio,
                "ft_boilerplate_count": structure.boilerplate_sentence_count,
                "ft_boilerplate_density": structure.boilerplate_density,
                "ft_ai_vocabulary_diversity": ai_full.vocabulary_diversity,
                "ft_ai_sentence_uniformity": ai_full.sentence_length_uniformity,
                "ft_ai_discourse_density": ai_full.discourse_marker_density,
            })
        else:
            ft_features.append({
                "openalex_id": row.get("openalex_id"),
                "has_fulltext": False,
            })

    ft_df = pd.DataFrame(ft_features)
    feature_df = feature_df.merge(ft_df, on="openalex_id", how="left")
    feature_df["has_fulltext"] = feature_df["has_fulltext"].fillna(False)

    # Methods-section similarity (the highest-value full-text feature)
    # Mills reuse methods sections near-verbatim across batches
    logger.info("Computing methods-section similarity...")
    feature_df = add_methods_similarity_features(
        feature_df, fulltext_data, id_column="pmcid",
    )

    logger.info(
        f"Enriched features: {feature_df['has_fulltext'].sum()} papers with full text"
    )

    return feature_df
