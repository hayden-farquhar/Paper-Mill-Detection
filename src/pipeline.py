"""
Main pipeline orchestrator for paper mill detection.

Runs the full pipeline: collect -> detect -> classify -> analyse.
Can also run individual stages for incremental work.
"""

import logging
from pathlib import Path

import pandas as pd

from src.collect.openalex_collector import collect_medical_ai_papers
from src.collect.pmc_fetcher import fetch_batch
from src.collect.retraction_loader import load_retraction_watch, match_with_corpus
from src.detect.tortured_phrases import TorturedPhraseDetector
from src.classify.feature_builder import build_features_from_abstracts, build_features_with_fulltext
from src.classify.classifier import train_and_evaluate, predict_corpus
from src.classify.validator import validate_against_retractions, estimate_prevalence

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def stage_collect(
    max_papers: int = 1000,
    year_range: tuple[int, int] = (2018, 2025),
) -> pd.DataFrame:
    """Stage 1: Collect medical AI papers from OpenAlex."""
    logger.info("=" * 60)
    logger.info("STAGE 1: COLLECT")
    logger.info("=" * 60)

    output_dir = DATA_DIR / "openalex"
    df = collect_medical_ai_papers(
        year_range=year_range,
        max_papers=max_papers,
        output_dir=output_dir,
    )

    logger.info(f"Collected {len(df)} papers")
    return df


def stage_detect(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: Run detection features on the corpus."""
    logger.info("=" * 60)
    logger.info("STAGE 2: DETECT")
    logger.info("=" * 60)

    detector = TorturedPhraseDetector()
    feature_df = build_features_from_abstracts(corpus_df, detector)

    # Save features
    output_path = DATA_DIR / "analysis" / "features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")

    return feature_df


def stage_classify(
    feature_df: pd.DataFrame,
    retraction_path: Path = None,
    corpus_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Stage 3: Train classifier and predict on corpus."""
    logger.info("=" * 60)
    logger.info("STAGE 3: CLASSIFY")
    logger.info("=" * 60)

    # Load retraction data if available
    if retraction_path and retraction_path.exists():
        retraction_df = load_retraction_watch(retraction_path)
        if corpus_df is not None:
            corpus_df = match_with_corpus(retraction_df, corpus_df)
            # Add labels to feature df
            feature_df = feature_df.merge(
                corpus_df[["openalex_id", "is_retracted", "retraction_reason"]],
                on="openalex_id",
                how="left",
            )
            feature_df["is_retracted"] = feature_df["is_retracted"].fillna(False)

    if "is_retracted" in feature_df.columns and feature_df["is_retracted"].sum() > 0:
        # Train and evaluate
        results = train_and_evaluate(feature_df)
        logger.info(f"\n{results.summary()}")

        # Predict on full corpus
        feature_df = predict_corpus(
            feature_df,
            feature_df[feature_df["is_retracted"].notna()],
        )
    else:
        logger.warning(
            "No retraction labels available. Running detection features only "
            "(no supervised classifier)."
        )
        # Use unsupervised scoring as fallback
        feature_df["mill_probability"] = _unsupervised_score(feature_df)
        feature_df["mill_flag"] = feature_df["mill_probability"] >= 0.5

    # Save predictions
    output_path = DATA_DIR / "analysis" / "predictions.csv"
    feature_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    return feature_df


def _unsupervised_score(feature_df: pd.DataFrame) -> pd.Series:
    """Rule-based scoring when no supervised labels are available.

    Weights are informed by the research integrity literature and ordered
    by specificity (how reliably the signal identifies paper mills vs
    producing false positives on legitimate papers):

    Tier 1 — High specificity (0.45 total):
      Tortured phrases (0.20): near-pathognomonic (Cabanac et al. 2021)
      Methods similarity (0.15): near-verbatim reuse across "independent" labs
      Abstract similarity (0.10): batch production captured at abstract level

    Tier 2 — Moderate specificity (0.30 total):
      Author network density (0.10): insular author pools
      Boilerplate/template patterns (0.08): formulaic but not unique to mills
      Interaction features (0.07): AI + structural signals combined
      Author reuse rate (0.05): same names across many papers

    Tier 3 — Low specificity (0.15 total):
      AI text markers (0.05): ambiguous post-2023 (legitimate LLM use)
      Vocabulary diversity (0.05): suggestive but many false positives
      Discourse marker density (0.03): weak signal alone
      International collaboration (0.02): domestic-only papers slightly more common in mills

    Geographic features deliberately excluded — they are analysed separately
    via sensitivity analysis to avoid using country as a quality proxy.

    Weights sum to 0.90 maximum (not 1.0) because no paper triggers all
    signals simultaneously; this prevents score inflation.
    """
    scores = pd.Series(0.0, index=feature_df.index)

    # ── Tier 1: High-specificity signals (0.45) ──

    # Tortured phrases — synonym-substitution is near-unique to mills
    if "tortured_phrase_count" in feature_df.columns:
        scores += (feature_df["tortured_phrase_count"] > 0).astype(float) * 0.20

    # Methods-section similarity — mills reuse methods verbatim
    if "methods_sim_max" in feature_df.columns:
        scores += feature_df["methods_sim_max"].clip(0, 1) * 0.15

    # Abstract similarity — batch production at abstract level
    if "sim_max" in feature_df.columns:
        scores += feature_df["sim_max"].clip(0, 1) * 0.10

    # ── Tier 2: Moderate-specificity signals (0.30) ──

    # Author network: insular co-authorship cliques
    if "net_coauthor_density" in feature_df.columns:
        scores += feature_df["net_coauthor_density"].clip(0, 1) * 0.10

    # Boilerplate density
    if "boilerplate_density" in feature_df.columns:
        bp_norm = feature_df["boilerplate_density"].clip(upper=20) / 20
        scores += bp_norm * 0.08

    # Interaction: AI markers combined with structural red flags
    if "interaction_ai_x_tortured" in feature_df.columns:
        scores += feature_df["interaction_ai_x_tortured"].clip(0, 5) / 5 * 0.07

    # Author reuse across corpus papers
    if "net_author_reuse_rate" in feature_df.columns:
        scores += feature_df["net_author_reuse_rate"].clip(0, 1) * 0.05

    # ── Tier 3: Low-specificity signals (0.15) ──

    if "ai_sentence_uniformity" in feature_df.columns:
        scores += feature_df["ai_sentence_uniformity"].clip(upper=1) * 0.05

    if "ai_vocabulary_diversity" in feature_df.columns:
        scores += (1 - feature_df["ai_vocabulary_diversity"].clip(0, 1)) * 0.05

    if "ai_discourse_marker_density" in feature_df.columns:
        dm_norm = feature_df["ai_discourse_marker_density"].clip(upper=30) / 30
        scores += dm_norm * 0.03

    # Domestic-only papers are slightly more common in mill output
    if "geo_is_international" in feature_df.columns:
        scores += (1 - feature_df["geo_is_international"]) * 0.02

    return scores.clip(0, 1)


def stage_analyse(predictions_df: pd.DataFrame):
    """Stage 4: Run analyses and generate outputs."""
    logger.info("=" * 60)
    logger.info("STAGE 4: ANALYSE")
    logger.info("=" * 60)

    from src.analyse.prevalence import (
        compute_overall_prevalence,
        plot_prevalence_by_year,
        plot_prevalence_by_journal,
    )
    from src.analyse.temporal_trends import test_pre_post_chatgpt
    from src.analyse.journal_analysis import journal_summary, compare_oa_vs_subscription

    figures_dir = OUTPUT_DIR / "figures"
    tables_dir = OUTPUT_DIR / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Overall prevalence
    prev = compute_overall_prevalence(predictions_df)
    logger.info(f"Overall prevalence: {prev['prevalence_pct']}% "
                f"(95% CI: {prev['ci_lower_pct']}-{prev['ci_upper_pct']}%)")

    # Temporal trend
    if "publication_year" in predictions_df.columns:
        chatgpt = test_pre_post_chatgpt(predictions_df)
        logger.info(f"Pre-ChatGPT: {chatgpt.get('pre_chatgpt_rate', 'N/A')}%")
        logger.info(f"Post-ChatGPT: {chatgpt.get('post_chatgpt_rate', 'N/A')}%")

        plot_prevalence_by_year(
            predictions_df,
            output_path=figures_dir / "prevalence_by_year.png",
        )

    # Journal analysis
    if "journal_name" in predictions_df.columns:
        j_summary = journal_summary(predictions_df)
        j_summary.to_csv(tables_dir / "journal_summary.csv", index=False)

        plot_prevalence_by_journal(
            predictions_df,
            output_path=figures_dir / "prevalence_by_journal.png",
        )

    # OA comparison
    if "is_oa" in predictions_df.columns:
        oa_comp = compare_oa_vs_subscription(predictions_df)
        logger.info(f"OA prevalence: {oa_comp.get('oa_prevalence_pct', 'N/A')}%")
        logger.info(f"Non-OA prevalence: {oa_comp.get('non_oa_prevalence_pct', 'N/A')}%")

    # Geographic analysis (regional aggregation, not country-level)
    from src.analyse.geographic import (
        prevalence_by_region,
        prevalence_by_income_group,
        geographic_sensitivity_analysis,
    )

    region_prev = prevalence_by_region(predictions_df)
    if len(region_prev) > 0:
        region_prev.to_csv(tables_dir / "prevalence_by_region.csv", index=False)
        logger.info("Prevalence by WHO region:")
        for _, row in region_prev.iterrows():
            logger.info(
                f"  {row['who_region_name']}: {row['prevalence_pct']}% "
                f"(n={row['n_papers']})"
            )

    income_prev = prevalence_by_income_group(predictions_df)
    if len(income_prev) > 0:
        income_prev.to_csv(tables_dir / "prevalence_by_income.csv", index=False)

    # Geographic sensitivity analysis (if labels available)
    if "is_retracted" in predictions_df.columns and predictions_df["is_retracted"].sum() >= 3:
        geo_sensitivity = geographic_sensitivity_analysis(predictions_df)
        logger.info(f"Geographic sensitivity: {geo_sensitivity.get('interpretation', 'N/A')}")
        # Save result
        import json
        sensitivity_path = tables_dir / "geographic_sensitivity.json"
        with open(sensitivity_path, "w") as f:
            json.dump(geo_sensitivity, f, indent=2)

    # Author pool detection
    from src.detect.author_network import AuthorNetworkAnalyser
    network = AuthorNetworkAnalyser()
    network.build_network(predictions_df)
    pools = network.find_author_pools(min_shared_papers=3, min_pool_size=3)
    if pools:
        pools_df = pd.DataFrame(pools)
        pools_df.to_csv(tables_dir / "author_pools.csv", index=False)
        logger.info(f"Found {len(pools)} potential author pools")
        for i, pool in enumerate(pools[:5]):
            logger.info(
                f"  Pool {i+1}: {pool['size']} authors, "
                f"{pool['n_papers']} papers, density={pool['density']}"
            )


def run_full_pipeline(
    max_papers: int = 1000,
    year_range: tuple[int, int] = (2018, 2025),
    retraction_path: Path = None,
):
    """Run the complete pipeline end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Stage 1: Collect
    corpus_df = stage_collect(max_papers=max_papers, year_range=year_range)

    # Stage 2: Detect
    feature_df = stage_detect(corpus_df)

    # Stage 3: Classify
    predictions_df = stage_classify(feature_df, retraction_path, corpus_df)

    # Stage 4: Analyse
    stage_analyse(predictions_df)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return predictions_df


if __name__ == "__main__":
    run_full_pipeline(max_papers=100)  # Small test run
