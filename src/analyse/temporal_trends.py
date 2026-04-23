"""
Temporal trend analysis for paper mill detection.

Examines how paper mill characteristics change over time,
with particular focus on the post-ChatGPT period (2023+).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_temporal_features(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
) -> pd.DataFrame:
    """Compute per-year summary statistics of detection features.

    Returns a DataFrame with one row per year and columns for
    mean values of each detection feature.
    """
    year_col = "publication_year"
    if year_col not in df.columns:
        logger.warning("No publication_year column")
        return pd.DataFrame()

    # Select numeric feature columns
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != year_col
    ]

    yearly = df.groupby(year_col)[feature_cols].mean().reset_index()
    return yearly


def test_pre_post_chatgpt(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
    cutoff_year: int = 2023,
) -> dict:
    """Compare prevalence before and after ChatGPT release.

    Simple two-sample proportion test (pre-2023 vs 2023+).
    """
    df = df.copy()
    df["flagged"] = (df[prob_col] >= threshold).astype(int)

    pre = df[df["publication_year"] < cutoff_year]
    post = df[df["publication_year"] >= cutoff_year]

    if len(pre) == 0 or len(post) == 0:
        return {"error": "Insufficient data for pre/post comparison"}

    pre_rate = pre["flagged"].mean()
    post_rate = post["flagged"].mean()

    # Two-proportion z-test
    n1, n2 = len(pre), len(post)
    p1, p2 = pre_rate, post_rate
    p_pooled = (pre["flagged"].sum() + post["flagged"].sum()) / (n1 + n2)

    if p_pooled == 0 or p_pooled == 1:
        z_stat = 0.0
    else:
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z_stat = (p2 - p1) / se if se > 0 else 0.0

    return {
        "pre_chatgpt_rate": round(pre_rate * 100, 2),
        "post_chatgpt_rate": round(post_rate * 100, 2),
        "pre_n": n1,
        "post_n": n2,
        "absolute_change": round((post_rate - pre_rate) * 100, 2),
        "z_statistic": round(z_stat, 3),
        "cutoff_year": cutoff_year,
    }


def plot_feature_trends(
    df: pd.DataFrame,
    features: list[str],
    output_path: Optional[Path] = None,
):
    """Plot how specific detection features change over time."""
    yearly = compute_temporal_features(df)

    if yearly.empty:
        return None

    n_features = len(features)
    fig, axes = plt.subplots(
        n_features, 1, figsize=(10, 3 * n_features), sharex=True
    )
    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        if feat in yearly.columns:
            ax.plot(yearly["publication_year"], yearly[feat], "o-", linewidth=2)
            ax.set_ylabel(feat.replace("_", " ").title())
            ax.axvline(x=2022.5, color="red", linestyle="--", alpha=0.5, label="ChatGPT release")
            ax.legend()

    axes[-1].set_xlabel("Publication Year")
    fig.suptitle("Detection Feature Trends Over Time", fontsize=14)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
