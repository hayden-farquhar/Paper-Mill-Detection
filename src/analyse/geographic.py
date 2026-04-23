"""
Geographic analysis with sensitivity safeguards.

Country-level paper mill data is scientifically relevant but editorially
sensitive. This module provides:

1. Regional aggregation — reporting by WHO/World Bank region rather than
   individual countries to avoid stigmatising specific nations
2. Contextualisation — normalising flagged rates against total publication
   volume from each region (high absolute counts from countries that publish
   more is expected, not evidence of higher mill rates)
3. Sensitivity analysis — demonstrating that classifier performance holds
   with and without geographic features, proving the model doesn't use
   country as a proxy for quality
4. Incentive context — framing geographic patterns in terms of structural
   factors (publish-or-perish pressure, institutional oversight capacity)

References:
  Else, H. (2022). How a torrent of COVID science changed research publishing
  — in seven charts. Nature.
  Shen, C., & Björk, B. (2015). 'Predatory' open access: a longitudinal
  study of article volumes and market characteristics. BMC Medicine.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# WHO region mapping for responsible geographic aggregation
# ISO 3166-1 alpha-2 -> WHO region
COUNTRY_TO_REGION = {
    # African Region (AFR)
    "DZ": "AFR", "AO": "AFR", "BJ": "AFR", "BW": "AFR", "BF": "AFR",
    "BI": "AFR", "CV": "AFR", "CM": "AFR", "CF": "AFR", "TD": "AFR",
    "KM": "AFR", "CG": "AFR", "CD": "AFR", "CI": "AFR", "GQ": "AFR",
    "ER": "AFR", "SZ": "AFR", "ET": "AFR", "GA": "AFR", "GM": "AFR",
    "GH": "AFR", "GN": "AFR", "GW": "AFR", "KE": "AFR", "LS": "AFR",
    "LR": "AFR", "MG": "AFR", "MW": "AFR", "ML": "AFR", "MR": "AFR",
    "MU": "AFR", "MZ": "AFR", "NA": "AFR", "NE": "AFR", "NG": "AFR",
    "RW": "AFR", "ST": "AFR", "SN": "AFR", "SC": "AFR", "SL": "AFR",
    "ZA": "AFR", "SS": "AFR", "TG": "AFR", "UG": "AFR", "TZ": "AFR",
    "ZM": "AFR", "ZW": "AFR",
    # Region of the Americas (AMR)
    "AG": "AMR", "AR": "AMR", "BS": "AMR", "BB": "AMR", "BZ": "AMR",
    "BO": "AMR", "BR": "AMR", "CA": "AMR", "CL": "AMR", "CO": "AMR",
    "CR": "AMR", "CU": "AMR", "DM": "AMR", "DO": "AMR", "EC": "AMR",
    "SV": "AMR", "GD": "AMR", "GT": "AMR", "GY": "AMR", "HT": "AMR",
    "HN": "AMR", "JM": "AMR", "MX": "AMR", "NI": "AMR", "PA": "AMR",
    "PY": "AMR", "PE": "AMR", "KN": "AMR", "LC": "AMR", "VC": "AMR",
    "SR": "AMR", "TT": "AMR", "US": "AMR", "UY": "AMR", "VE": "AMR",
    # South-East Asia Region (SEAR)
    "BD": "SEAR", "BT": "SEAR", "KP": "SEAR", "IN": "SEAR", "ID": "SEAR",
    "MV": "SEAR", "MM": "SEAR", "NP": "SEAR", "LK": "SEAR", "TH": "SEAR",
    "TL": "SEAR",
    # European Region (EUR)
    "AL": "EUR", "AD": "EUR", "AM": "EUR", "AT": "EUR", "AZ": "EUR",
    "BY": "EUR", "BE": "EUR", "BA": "EUR", "BG": "EUR", "HR": "EUR",
    "CY": "EUR", "CZ": "EUR", "DK": "EUR", "EE": "EUR", "FI": "EUR",
    "FR": "EUR", "GE": "EUR", "DE": "EUR", "GR": "EUR", "HU": "EUR",
    "IS": "EUR", "IE": "EUR", "IL": "EUR", "IT": "EUR", "KZ": "EUR",
    "KG": "EUR", "LV": "EUR", "LT": "EUR", "LU": "EUR", "MT": "EUR",
    "MC": "EUR", "ME": "EUR", "NL": "EUR", "MK": "EUR", "NO": "EUR",
    "PL": "EUR", "PT": "EUR", "MD": "EUR", "RO": "EUR", "RU": "EUR",
    "SM": "EUR", "RS": "EUR", "SK": "EUR", "SI": "EUR", "ES": "EUR",
    "SE": "EUR", "CH": "EUR", "TJ": "EUR", "TR": "EUR", "TM": "EUR",
    "UA": "EUR", "GB": "EUR", "UZ": "EUR",
    # Eastern Mediterranean Region (EMR)
    "AF": "EMR", "BH": "EMR", "DJ": "EMR", "EG": "EMR", "IR": "EMR",
    "IQ": "EMR", "JO": "EMR", "KW": "EMR", "LB": "EMR", "LY": "EMR",
    "MA": "EMR", "OM": "EMR", "PK": "EMR", "PS": "EMR", "QA": "EMR",
    "SA": "EMR", "SO": "EMR", "SD": "EMR", "SY": "EMR", "TN": "EMR",
    "AE": "EMR", "YE": "EMR",
    # Western Pacific Region (WPR)
    "AU": "WPR", "BN": "WPR", "KH": "WPR", "CN": "WPR", "CK": "WPR",
    "FJ": "WPR", "JP": "WPR", "KR": "WPR", "LA": "WPR", "MY": "WPR",
    "MH": "WPR", "FM": "WPR", "MN": "WPR", "NR": "WPR", "NZ": "WPR",
    "NU": "WPR", "PW": "WPR", "PG": "WPR", "PH": "WPR", "WS": "WPR",
    "SG": "WPR", "SB": "WPR", "TO": "WPR", "TV": "WPR", "VU": "WPR",
    "VN": "WPR", "TW": "WPR",
}

REGION_NAMES = {
    "AFR": "African Region",
    "AMR": "Region of the Americas",
    "SEAR": "South-East Asia Region",
    "EUR": "European Region",
    "EMR": "Eastern Mediterranean Region",
    "WPR": "Western Pacific Region",
}

# World Bank income groups for additional context
INCOME_GROUPS = {
    "HIC": "High-income countries",
    "UMIC": "Upper-middle-income countries",
    "LMIC": "Lower-middle-income countries",
    "LIC": "Low-income countries",
}

# Simplified income classification (selected major research-producing countries)
COUNTRY_TO_INCOME = {
    "US": "HIC", "GB": "HIC", "DE": "HIC", "JP": "HIC", "AU": "HIC",
    "CA": "HIC", "FR": "HIC", "IT": "HIC", "KR": "HIC", "NL": "HIC",
    "SE": "HIC", "CH": "HIC", "DK": "HIC", "NO": "HIC", "FI": "HIC",
    "SG": "HIC", "IL": "HIC", "NZ": "HIC", "AT": "HIC", "BE": "HIC",
    "IE": "HIC", "ES": "HIC", "PT": "HIC", "CZ": "HIC", "PL": "HIC",
    "SA": "HIC", "AE": "HIC", "QA": "HIC", "KW": "HIC", "BH": "HIC",
    "TW": "HIC",
    "CN": "UMIC", "BR": "UMIC", "MX": "UMIC", "TR": "UMIC", "MY": "UMIC",
    "TH": "UMIC", "ZA": "UMIC", "CO": "UMIC", "RU": "UMIC", "AR": "UMIC",
    "IR": "UMIC", "IQ": "UMIC", "RO": "UMIC", "KZ": "UMIC",
    "IN": "LMIC", "PK": "LMIC", "BD": "LMIC", "VN": "LMIC", "EG": "LMIC",
    "NG": "LMIC", "PH": "LMIC", "ID": "LMIC", "KE": "LMIC", "GH": "LMIC",
    "NP": "LMIC", "LK": "LMIC", "MA": "LMIC", "TN": "LMIC", "MM": "LMIC",
    "ET": "LMIC",
}


def map_country_to_region(country_code: str) -> str:
    """Map ISO country code to WHO region."""
    return COUNTRY_TO_REGION.get(country_code.upper().strip(), "Unknown")


def map_country_to_income(country_code: str) -> str:
    """Map ISO country code to World Bank income group."""
    return COUNTRY_TO_INCOME.get(country_code.upper().strip(), "Unknown")


def add_geographic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add WHO region and income group columns to corpus DataFrame.

    Uses the 'corresponding_countries' or 'all_countries' column.
    For papers with multiple countries, uses the first listed
    (typically the corresponding author's country).
    """
    df = df.copy()

    # Get primary country (first in the list)
    country_col = None
    for candidate in ["corresponding_countries", "all_countries"]:
        if candidate in df.columns:
            country_col = candidate
            break

    if country_col is None:
        df["primary_country"] = "Unknown"
        df["who_region"] = "Unknown"
        df["income_group"] = "Unknown"
        return df

    df["primary_country"] = (
        df[country_col]
        .fillna("")
        .str.split(";")
        .str[0]
        .str.strip()
    )

    df["who_region"] = df["primary_country"].apply(map_country_to_region)
    df["who_region_name"] = df["who_region"].map(REGION_NAMES).fillna("Unknown")
    df["income_group"] = df["primary_country"].apply(map_country_to_income)

    return df


def prevalence_by_region(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
    min_papers: int = 20,
) -> pd.DataFrame:
    """Compute flagged-paper prevalence by WHO region.

    Reports regional aggregates rather than individual countries.
    Includes publication volume context to distinguish high absolute
    counts (expected from prolific regions) from genuinely elevated rates.
    """
    df = add_geographic_columns(df) if "who_region" not in df.columns else df.copy()
    df["flagged"] = (df[prob_col] >= threshold).astype(int)

    grouped = df.groupby(["who_region", "who_region_name"]).agg(
        n_papers=("flagged", "count"),
        n_flagged=("flagged", "sum"),
        mean_prob=(prob_col, "mean"),
        publication_share=("flagged", "count"),
    ).reset_index()

    total_papers = grouped["n_papers"].sum()
    grouped["publication_share"] = (grouped["n_papers"] / total_papers * 100).round(2)
    grouped["prevalence_pct"] = (grouped["n_flagged"] / grouped["n_papers"] * 100).round(2)

    # Filter to regions with enough papers for stable estimates
    grouped = grouped[grouped["n_papers"] >= min_papers]

    return grouped.sort_values("prevalence_pct", ascending=False)


def prevalence_by_income_group(
    df: pd.DataFrame,
    prob_col: str = "mill_probability",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute prevalence by World Bank income group."""
    df = add_geographic_columns(df) if "income_group" not in df.columns else df.copy()
    df["flagged"] = (df[prob_col] >= threshold).astype(int)

    grouped = df.groupby("income_group").agg(
        n_papers=("flagged", "count"),
        n_flagged=("flagged", "sum"),
        mean_prob=(prob_col, "mean"),
    ).reset_index()

    grouped["prevalence_pct"] = (grouped["n_flagged"] / grouped["n_papers"] * 100).round(2)
    grouped = grouped[grouped["income_group"] != "Unknown"]

    return grouped.sort_values("prevalence_pct", ascending=False)


def geographic_sensitivity_analysis(
    feature_df: pd.DataFrame,
    label_column: str = "is_retracted",
    geographic_features: Optional[list[str]] = None,
) -> dict:
    """Test whether the classifier relies on geography as a proxy.

    Trains the classifier twice — with and without geographic features —
    and compares performance. If performance is similar, geographic features
    aren't driving predictions (good). If performance drops substantially
    without them, the classifier may be using country as a quality proxy
    (editorially problematic — should be disclosed).

    Args:
        feature_df: Feature DataFrame with labels
        label_column: Binary label column
        geographic_features: Columns to exclude in the sensitivity test.
            If None, auto-detects country/region columns.

    Returns:
        Dict comparing with/without geographic features
    """
    from src.classify.classifier import train_and_evaluate

    if label_column not in feature_df.columns or feature_df[label_column].sum() < 3:
        return {"error": "Insufficient labels for sensitivity analysis"}

    # Identify geographic feature columns
    if geographic_features is None:
        geographic_features = [
            c for c in feature_df.columns
            if any(kw in c.lower() for kw in [
                "country", "region", "income", "geo_",
            ])
        ]

    if not geographic_features:
        return {
            "note": "No geographic features found in feature set — "
                    "sensitivity analysis not needed"
        }

    # Model WITH geographic features
    logger.info("Training classifier WITH geographic features...")
    results_with = train_and_evaluate(feature_df, label_column=label_column)

    # Model WITHOUT geographic features
    logger.info("Training classifier WITHOUT geographic features...")
    non_geo_cols = [c for c in feature_df.columns if c not in geographic_features]
    feature_df_no_geo = feature_df[non_geo_cols]
    results_without = train_and_evaluate(feature_df_no_geo, label_column=label_column)

    auc_diff = results_with.auc_roc - results_without.auc_roc
    ap_diff = results_with.average_precision - results_without.average_precision

    # Interpret
    if abs(ap_diff) < 0.02:
        interpretation = (
            "Geographic features have negligible impact on classification. "
            "The model does not rely on country as a proxy for paper quality."
        )
    elif ap_diff > 0.05:
        interpretation = (
            "CAUTION: Geographic features meaningfully improve classification "
            f"(AP delta: +{ap_diff:.3f}). This suggests the model may use "
            "country of origin as a proxy. Consider reporting results with "
            "and without geographic features, and contextualising any "
            "geographic patterns with structural factors."
        )
    else:
        interpretation = (
            f"Geographic features have minor impact (AP delta: {ap_diff:+.3f}). "
            "The model primarily relies on non-geographic signals."
        )

    result = {
        "with_geography": {
            "auc_roc": round(results_with.auc_roc, 4),
            "average_precision": round(results_with.average_precision, 4),
        },
        "without_geography": {
            "auc_roc": round(results_without.auc_roc, 4),
            "average_precision": round(results_without.average_precision, 4),
        },
        "auc_delta": round(auc_diff, 4),
        "ap_delta": round(ap_diff, 4),
        "geographic_features_tested": geographic_features,
        "interpretation": interpretation,
    }

    logger.info(f"Geographic sensitivity: {interpretation}")

    return result


def prepare_geographic_features(
    corpus_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare geographic features for the classifier.

    Encodes geography at the REGIONAL level (not country level) to:
    1. Reduce dimensionality
    2. Avoid learning country-specific biases
    3. Maintain analytical utility

    Returns DataFrame with one-hot encoded regional features.
    """
    df = add_geographic_columns(corpus_df)

    # One-hot encode WHO regions (not countries)
    region_dummies = pd.get_dummies(
        df["who_region"],
        prefix="geo_region",
        dtype=float,
    )

    # Also encode income group
    income_dummies = pd.get_dummies(
        df["income_group"],
        prefix="geo_income",
        dtype=float,
    )

    # Multi-country flag (international collaborations may be lower risk)
    if "all_countries" in df.columns:
        df["geo_n_countries"] = (
            df["all_countries"]
            .fillna("")
            .str.split(";")
            .apply(lambda x: len([c for c in x if c.strip()]))
        )
        df["geo_is_international"] = (df["geo_n_countries"] > 1).astype(float)
    else:
        df["geo_n_countries"] = 0
        df["geo_is_international"] = 0.0

    result = pd.concat([
        df[["openalex_id", "geo_n_countries", "geo_is_international"]],
        region_dummies,
        income_dummies,
    ], axis=1)

    return result
