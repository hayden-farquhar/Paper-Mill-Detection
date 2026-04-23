"""
Leave-Hindawi-out sensitivity analysis.

Post-hoc analysis added 2026-04-24 to address the dominance of
Hindawi-retracted papers (85% of positive labels when broadly defined).

Removes all papers from known Hindawi/Wiley-retracted journals and
re-runs the classifier with identical architecture and hyperparameters.

Input: data/analysis/predictions_primary.csv
Output: data/analysis/sensitivity_leave_hindawi_out.json

This script uses the same classifier architecture and hyperparameters
as pre-registered (see OSF DOI: 10.17605/OSF.IO/JB4T6).
"""

import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- Configuration ---

HINDAWI_JOURNAL_KEYWORDS = [
    "Computational Intelligence and Neuroscience",
    "Security and Communication Networks",
    "Wireless Communications and Mobile Computing",
    "Journal of Healthcare Engineering",
    "Journal of Sensors",
    "Contrast Media",
    "BioMed Research International",
    "Computational and Mathematical Methods in Medicine",
    "Scientific Programming",
    "Mobile Information Systems",
    "Journal of Intelligent",
    "Disease Markers",
    "Evidence-Based Complementary",
    "Oxidative Medicine",
    "Complexity",
    "Mathematical Problems in Engineering",
    "Journal of Environmental and Public Health",
    "Applied Bionics and Biomechanics",
    "Journal of Food Quality",
    "International Transactions on Electrical Energy Systems",
    "Discrete Dynamics in Nature and Society",
]

EXCLUDE_COLUMNS = {
    "openalex_id", "doi", "pmid", "pmcid", "is_retracted",
    "retraction_reason", "title", "abstract", "label",
    "publication_year", "journal_name", "is_oa", "oa_status",
    "mill_probability_raw", "mill_probability", "mill_flag",
    "pu_label_frequency", "cited_by_count",
    "retraction_reasons", "doi_norm", "retraction_subtype",
    "geo_region_AFR", "geo_region_AMR", "geo_region_EMR",
    "geo_region_EUR", "geo_region_SEAR", "geo_region_Unknown",
    "geo_region_WPR", "geo_income_HIC", "geo_income_LMIC",
    "geo_income_UMIC", "geo_income_Unknown", "cluster_label",
}

INPUT_PATH = "data/analysis/predictions_primary.csv"
OUTPUT_PATH = "data/analysis/sensitivity_leave_hindawi_out.json"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Identify Hindawi journals
    pattern = "|".join(HINDAWI_JOURNAL_KEYWORDS)
    hindawi_mask = df["journal_name"].str.contains(
        pattern, case=False, na=False
    )

    df_no_hindawi = df[~hindawi_mask].copy()

    print(f"Original corpus: {len(df)} papers ({df['is_retracted'].sum()} retracted)")
    print(f"Hindawi removed: {hindawi_mask.sum()} papers")
    print(f"Remaining: {len(df_no_hindawi)} papers ({df_no_hindawi['is_retracted'].sum()} retracted)")

    # Prepare features
    feature_cols = [
        c for c in df_no_hindawi.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_COLUMNS
    ]
    X = df_no_hindawi[feature_cols].fillna(0).values
    y = df_no_hindawi["is_retracted"].astype(int).values
    n_pos, n_neg = y.sum(), len(y) - y.sum()

    # Build ensemble (identical to pre-registered specification)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, eval_metric="logloss",
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)], voting="soft"
    )

    # Cross-validate
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probas = cross_val_predict(
        ensemble, X_scaled, y, cv=cv, method="predict_proba"
    )[:, 1]

    # Metrics
    ap = average_precision_score(y, probas)
    auroc = roc_auc_score(y, probas)

    # PU learning correction (e3 method)
    ensemble_fitted = clone(ensemble)
    ensemble_fitted.fit(X_scaled, y)
    pos_probas = probas[y == 1]
    sorted_pos = np.sort(pos_probas)[::-1]
    top_half = sorted_pos[: max(len(sorted_pos) // 2, 1)]
    c = min(top_half.mean(), 1.0)

    # Prevalence on unlabelled only
    unlabelled_probas = probas[y == 0]
    corrected_probas = np.clip(unlabelled_probas / c, 0, 1)
    prev = corrected_probas.mean() * 100

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_prevs = []
    for _ in range(2000):
        idx = rng.choice(len(corrected_probas), size=len(corrected_probas), replace=True)
        boot_prevs.append(corrected_probas[idx].mean() * 100)
    ci_lo, ci_hi = np.percentile(boot_prevs, 2.5), np.percentile(boot_prevs, 97.5)

    # Feature importances
    rf_imp = ensemble_fitted.named_estimators_["rf"].feature_importances_
    xgb_imp = ensemble_fitted.named_estimators_["xgb"].feature_importances_
    avg_imp = (rf_imp + xgb_imp) / 2
    top_features = sorted(zip(feature_cols, avg_imp), key=lambda x: -x[1])[:10]

    # Save results
    results = {
        "analysis": "leave_hindawi_out",
        "original_n": len(df),
        "original_retracted": int(df["is_retracted"].sum()),
        "hindawi_removed": int(hindawi_mask.sum()),
        "hindawi_retracted_removed": int((hindawi_mask & df["is_retracted"]).sum()),
        "remaining_n": len(df_no_hindawi),
        "remaining_retracted": int(df_no_hindawi["is_retracted"].sum()),
        "ap": round(ap, 3),
        "auroc": round(auroc, 3),
        "pu_c": round(c, 3),
        "prevalence_pct": round(prev, 1),
        "ci_lower_pct": round(ci_lo, 1),
        "ci_upper_pct": round(ci_hi, 1),
        "top_features": [
            {"feature": f, "importance": round(i, 4)} for f, i in top_features
        ],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAP: {ap:.3f}")
    print(f"AUC-ROC: {auroc:.3f}")
    print(f"PU c: {c:.3f}")
    print(f"Prevalence: {prev:.1f}% (95% CI: {ci_lo:.1f}–{ci_hi:.1f}%)")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
