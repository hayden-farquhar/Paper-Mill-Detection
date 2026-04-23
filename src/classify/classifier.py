"""
Ensemble classifier for paper mill detection.

Combines Random Forest and XGBoost in a soft-voting ensemble.
Handles class imbalance (retracted papers are rare) via SMOTE
or class weighting.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class ClassifierResults:
    """Results from classifier training and evaluation."""

    auc_roc: float = 0.0
    average_precision: float = 0.0
    classification_report: str = ""
    feature_importances: dict = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None

    def summary(self) -> str:
        return (
            f"AUC-ROC: {self.auc_roc:.3f}\n"
            f"Average Precision: {self.average_precision:.3f}\n"
            f"\n{self.classification_report}"
        )


def prepare_features(
    feature_df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Prepare feature matrix from DataFrame.

    Handles missing values and selects numeric columns.

    Returns:
        (feature_matrix, column_names)
    """
    if feature_columns:
        # Use only specified columns that exist
        available = [c for c in feature_columns if c in feature_df.columns]
    else:
        # Use all numeric columns except IDs and labels
        exclude = {"openalex_id", "doi", "pmid", "pmcid", "is_retracted",
                    "retraction_reason", "title", "abstract", "label"}
        available = [
            c for c in feature_df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    X = feature_df[available].copy()

    # Convert booleans to int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # Fill NaN with 0 (missing features)
    X = X.fillna(0)

    return X.values, list(X.columns)


def build_ensemble(
    class_weight: str = "balanced",
    n_positive: int = 50,
    n_negative: int = 950,
) -> VotingClassifier:
    """Build the RF + XGBoost voting ensemble.

    The scale_pos_weight for XGBoost is computed dynamically from the
    actual class ratio rather than hardcoded, because the imbalance
    depends on how many retracted papers we match.
    """
    imbalance_ratio = max(n_negative / max(n_positive, 1), 1.0)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric="logloss",
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft",
    )

    return ensemble


def train_and_evaluate(
    feature_df: pd.DataFrame,
    label_column: str = "is_retracted",
    feature_columns: Optional[list[str]] = None,
    n_folds: int = 5,
) -> ClassifierResults:
    """Train ensemble classifier with cross-validation.

    Args:
        feature_df: DataFrame with features and labels
        label_column: Column name for binary labels
        feature_columns: Specific feature columns to use
        n_folds: Number of cross-validation folds

    Returns:
        ClassifierResults with evaluation metrics
    """
    X, col_names = prepare_features(feature_df, feature_columns)
    y = feature_df[label_column].astype(int).values

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    logger.info(f"Training: {n_pos} positive, {n_neg} negative samples")

    if n_pos < 5:
        logger.warning(
            f"Very few positive samples ({n_pos}). Results may be unreliable."
        )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated predictions
    ensemble = build_ensemble(n_positive=n_pos, n_negative=n_neg)
    cv = StratifiedKFold(n_splits=min(n_folds, n_pos), shuffle=True, random_state=42)

    probas = cross_val_predict(
        ensemble, X_scaled, y, cv=cv, method="predict_proba"
    )[:, 1]

    # Find optimal threshold via F1 on the PR curve (0.5 is usually wrong
    # under severe class imbalance — prevalence-dependent)
    precisions, recalls, pr_thresholds = precision_recall_curve(y, probas)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )
    best_idx = f1_scores.argmax()
    optimal_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5

    preds = (probas >= optimal_threshold).astype(int)

    # Metrics — average precision (PR-AUC) is the primary metric under
    # class imbalance; AUC-ROC is reported but can be misleadingly high
    results = ClassifierResults()
    results.auc_roc = roc_auc_score(y, probas)
    results.average_precision = average_precision_score(y, probas)
    results.classification_report = classification_report(
        y, preds, target_names=["legitimate", "paper_mill"], zero_division=0
    )

    logger.info(f"Optimal threshold (max F1 on PR curve): {optimal_threshold:.3f}")
    results.predictions = preds
    results.probabilities = probas

    # Feature importances (train on full data)
    ensemble.fit(X_scaled, y)
    rf_importance = ensemble.named_estimators_["rf"].feature_importances_
    xgb_importance = ensemble.named_estimators_["xgb"].feature_importances_
    avg_importance = (rf_importance + xgb_importance) / 2

    results.feature_importances = dict(
        sorted(zip(col_names, avg_importance), key=lambda x: -x[1])
    )

    logger.info(f"AUC-ROC: {results.auc_roc:.3f}")
    logger.info(f"Average Precision: {results.average_precision:.3f}")

    return results


def predict_corpus(
    feature_df: pd.DataFrame,
    train_feature_df: pd.DataFrame,
    label_column: str = "is_retracted",
    feature_columns: Optional[list[str]] = None,
    use_pu_learning: bool = True,
) -> pd.DataFrame:
    """Train on labelled data and predict on full corpus.

    When use_pu_learning=True (default), applies Positive-Unlabelled
    correction to account for undetected mill papers in the unlabelled set.

    Args:
        feature_df: Full corpus features (to predict)
        train_feature_df: Labelled subset for training
        label_column: Label column in train_feature_df
        feature_columns: Feature columns to use
        use_pu_learning: Apply PU correction to probabilities

    Returns:
        feature_df with added 'mill_probability', 'mill_probability_raw',
        and 'mill_flag' columns
    """
    X_train, col_names = prepare_features(train_feature_df, feature_columns)
    y_train = train_feature_df[label_column].astype(int).values

    X_full, _ = prepare_features(feature_df, feature_columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    ensemble = build_ensemble(n_positive=n_pos, n_negative=n_neg)
    ensemble.fit(X_train_scaled, y_train)

    raw_probas = ensemble.predict_proba(X_full_scaled)[:, 1]

    result_df = feature_df.copy()
    result_df["mill_probability_raw"] = raw_probas

    if use_pu_learning and n_pos >= 3:
        from src.classify.pu_learning import pu_classify
        from sklearn.base import clone

        logger.info("Applying PU learning correction...")
        pu_result = pu_classify(
            X_train_scaled, y_train,
            clone(ensemble),
            method="e3",  # Robust estimator
            use_cv=True,
        )
        logger.info(pu_result.summary())

        # Apply the learned label frequency to correct full-corpus predictions
        from src.classify.pu_learning import correct_probabilities
        corrected_probas = correct_probabilities(raw_probas, pu_result.label_frequency)

        result_df["mill_probability"] = corrected_probas
        result_df["pu_label_frequency"] = pu_result.label_frequency
    else:
        result_df["mill_probability"] = raw_probas

    result_df["mill_flag"] = result_df["mill_probability"] >= 0.5

    logger.info(
        f"Flagged {result_df['mill_flag'].sum()} / {len(result_df)} papers "
        f"({result_df['mill_flag'].mean()*100:.1f}%)"
    )

    return result_df
