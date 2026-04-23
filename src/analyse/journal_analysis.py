"""
Journal-level analysis of paper mill prevalence.

Examines which journals are most affected, broken down by
journal tier, open access status, and subject area.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def journal_summary(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
    min_papers: int = 5,
) -> pd.DataFrame:
    """Compute per-journal summary statistics.

    Returns DataFrame sorted by prevalence with columns:
    journal_name, n_papers, n_flagged, prevalence_pct, mean_prob,
    mean_citations, oa_fraction.
    """
    df = df.copy()
    df["flagged"] = (df[prob_col] >= threshold).astype(int)

    grouped = df.groupby("journal_name").agg(
        n_papers=("flagged", "count"),
        n_flagged=("flagged", "sum"),
        mean_prob=(prob_col, "mean"),
        mean_citations=("cited_by_count", "mean"),
        oa_fraction=("is_oa", "mean"),
    ).reset_index()

    grouped["prevalence_pct"] = (grouped["n_flagged"] / grouped["n_papers"] * 100).round(2)
    grouped = grouped[grouped["n_papers"] >= min_papers]

    return grouped.sort_values("prevalence_pct", ascending=False)


def compare_oa_vs_subscription(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
) -> dict:
    """Compare paper mill prevalence between OA and subscription journals."""
    df = df.copy()
    df["flagged"] = (df[prob_col] >= threshold).astype(int)

    oa = df[df["is_oa"] == True]
    non_oa = df[df["is_oa"] == False]

    return {
        "oa_prevalence_pct": round(oa["flagged"].mean() * 100, 2) if len(oa) > 0 else None,
        "non_oa_prevalence_pct": round(non_oa["flagged"].mean() * 100, 2) if len(non_oa) > 0 else None,
        "oa_n": len(oa),
        "non_oa_n": len(non_oa),
        "oa_mean_prob": round(oa[prob_col].mean(), 4) if len(oa) > 0 else None,
        "non_oa_mean_prob": round(non_oa[prob_col].mean(), 4) if len(non_oa) > 0 else None,
    }


def prevalence_by_citation_quartile(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute prevalence by citation count quartile.

    Tests whether less-cited papers are more likely to be flagged.
    """
    df = df.copy()
    df["flagged"] = (df[prob_col] >= threshold).astype(int)

    df["citation_quartile"] = pd.qcut(
        df["cited_by_count"].clip(lower=0),
        q=4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
        duplicates="drop",
    )

    result = df.groupby("citation_quartile").agg(
        n_papers=("flagged", "count"),
        n_flagged=("flagged", "sum"),
        mean_prob=(prob_col, "mean"),
    ).reset_index()

    result["prevalence_pct"] = (result["n_flagged"] / result["n_papers"] * 100).round(2)

    return result
