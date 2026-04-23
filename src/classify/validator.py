"""
Validate classifier against Retraction Watch data and compute prevalence.

Cross-references predictions with known retractions to assess
detection performance and estimate the true prevalence of
paper mill output in the medical AI literature.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)


def validate_against_retractions(
    predictions_df: pd.DataFrame,
    retraction_df: pd.DataFrame,
    prob_column: str = "mill_probability",
) -> dict:
    """Validate predictions against known retractions.

    Args:
        predictions_df: Corpus with 'mill_probability' column
        retraction_df: Retraction Watch records matched to corpus
        prob_column: Column with predicted probabilities

    Returns:
        Validation metrics dict
    """
    # Merge retraction labels
    if "is_retracted" not in predictions_df.columns:
        logger.warning("No 'is_retracted' column — run retraction_loader.match_with_corpus first")
        return {}

    labelled = predictions_df[predictions_df["is_retracted"].notna()].copy()
    y_true = labelled["is_retracted"].astype(int)
    y_prob = labelled[prob_column]

    if y_true.sum() == 0:
        logger.warning("No retracted papers in corpus — cannot validate")
        return {}

    # Metrics at various thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    threshold_results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        threshold_results.append({
            "threshold": t,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        })

    auc = roc_auc_score(y_true, y_prob)

    return {
        "auc_roc": auc,
        "n_retracted": int(y_true.sum()),
        "n_total": len(y_true),
        "threshold_results": threshold_results,
    }


def estimate_prevalence(
    predictions_df: pd.DataFrame,
    prob_column: str = "mill_probability",
    threshold: float = 0.5,
    bootstrap_n: int = 1000,
) -> dict:
    """Estimate prevalence of paper mill characteristics with confidence intervals.

    Uses bootstrap resampling for CI estimation.

    Args:
        predictions_df: Full corpus with predictions
        prob_column: Probability column
        threshold: Classification threshold
        bootstrap_n: Number of bootstrap iterations

    Returns:
        Dict with prevalence estimate and 95% CI
    """
    probs = predictions_df[prob_column].values
    flagged = (probs >= threshold).astype(int)

    point_estimate = flagged.mean()

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_estimates = []
    for _ in range(bootstrap_n):
        boot_sample = rng.choice(flagged, size=len(flagged), replace=True)
        boot_estimates.append(boot_sample.mean())

    ci_lower = np.percentile(boot_estimates, 2.5)
    ci_upper = np.percentile(boot_estimates, 97.5)

    return {
        "prevalence": round(point_estimate, 4),
        "prevalence_pct": round(point_estimate * 100, 2),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_lower_pct": round(ci_lower * 100, 2),
        "ci_upper_pct": round(ci_upper * 100, 2),
        "n_flagged": int(flagged.sum()),
        "n_total": len(flagged),
        "threshold": threshold,
    }


def prevalence_by_year(
    predictions_df: pd.DataFrame,
    prob_column: str = "mill_probability",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute prevalence broken down by publication year."""
    df = predictions_df.copy()
    df["flagged"] = (df[prob_column] >= threshold).astype(int)

    yearly = df.groupby("publication_year").agg(
        n_papers=("flagged", "count"),
        n_flagged=("flagged", "sum"),
        mean_probability=(prob_column, "mean"),
    ).reset_index()

    yearly["prevalence_pct"] = (yearly["n_flagged"] / yearly["n_papers"] * 100).round(2)

    return yearly
