"""
Prevalence analysis: estimate the rate of paper mill characteristics
in the medical AI literature, overall and by subgroup.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_overall_prevalence(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
) -> dict:
    """Compute overall prevalence with bootstrap CI."""
    flagged = (df[prob_col] >= threshold).astype(int)
    point = flagged.mean()

    rng = np.random.default_rng(42)
    boots = [
        rng.choice(flagged.values, size=len(flagged), replace=True).mean()
        for _ in range(2000)
    ]

    return {
        "prevalence_pct": round(point * 100, 2),
        "ci_lower_pct": round(np.percentile(boots, 2.5) * 100, 2),
        "ci_upper_pct": round(np.percentile(boots, 97.5) * 100, 2),
        "n_flagged": int(flagged.sum()),
        "n_total": len(flagged),
    }


def prevalence_by_group(
    df: pd.DataFrame,
    group_col: str,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Compute prevalence broken down by a grouping variable."""
    df = df.copy()
    df["flagged"] = (df[prob_col] >= threshold).astype(int)

    grouped = df.groupby(group_col).agg(
        n_papers=("flagged", "count"),
        n_flagged=("flagged", "sum"),
        mean_prob=(prob_col, "mean"),
    ).reset_index()

    grouped["prevalence_pct"] = (grouped["n_flagged"] / grouped["n_papers"] * 100).round(2)
    grouped = grouped[grouped["n_papers"] >= min_group_size]

    return grouped.sort_values("prevalence_pct", ascending=False)


def plot_prevalence_by_year(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
    output_path: Optional[Path] = None,
):
    """Plot temporal trend in paper mill prevalence."""
    yearly = prevalence_by_group(df, "publication_year", prob_col, threshold, min_group_size=5)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart of paper count
    ax1.bar(yearly["publication_year"], yearly["n_papers"], alpha=0.3, color="steelblue", label="Total papers")
    ax1.set_xlabel("Publication Year")
    ax1.set_ylabel("Number of Papers", color="steelblue")

    # Line chart of prevalence
    ax2 = ax1.twinx()
    ax2.plot(yearly["publication_year"], yearly["prevalence_pct"], "ro-", linewidth=2, label="Flagged %")
    ax2.set_ylabel("Flagged Papers (%)", color="red")

    plt.title("Paper Mill Characteristics in Medical AI Literature Over Time")
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_prevalence_by_journal(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
    top_n: int = 20,
    output_path: Optional[Path] = None,
):
    """Plot prevalence by journal (top N journals by paper count)."""
    journal_stats = prevalence_by_group(
        df, "journal_name", prob_col, threshold, min_group_size=10
    )

    # Top N by total papers
    top = journal_stats.nlargest(top_n, "n_papers")

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        range(len(top)),
        top["prevalence_pct"],
        color=["indianred" if p > 10 else "steelblue" for p in top["prevalence_pct"]],
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["journal_name"].str[:50])
    ax.set_xlabel("Flagged Papers (%)")
    ax.set_title(f"Paper Mill Prevalence by Journal (Top {top_n} by Volume)")

    # Add paper counts as annotations
    for i, (_, row) in enumerate(top.iterrows()):
        ax.annotate(
            f"n={row['n_papers']}",
            xy=(row["prevalence_pct"], i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
