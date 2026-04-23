"""
Positive-Unlabelled (PU) learning for paper mill detection.

Standard supervised classifiers assume all unlabelled papers are legitimate.
This is wrong — unlabelled papers include undetected mill output. PU learning
explicitly models this by:

1. Training a standard classifier treating unlabelled as negative
2. Estimating the "label frequency" c = P(labelled | positive) using
   predictions on the known positive set (Elkan & Noto, 2008)
3. Correcting predicted probabilities: P(positive | x) = P(s=1 | x) / c

This produces:
- More accurate prevalence estimates (corrected upward)
- Better-calibrated individual probabilities
- A principled estimate of the "dark figure" (undetected mill papers)

Reference:
  Elkan, C., & Noto, K. (2008). Learning classifiers from only positive
  and unlabeled data. KDD '08.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class PUResult:
    """Results from PU learning."""

    # Label frequency: P(labelled | truly positive)
    # Low c means most positives are unlabelled (many hidden mill papers)
    label_frequency: float = 1.0

    # Corrected probabilities for each paper
    corrected_probabilities: Optional[np.ndarray] = None

    # Corrected prevalence
    raw_prevalence: float = 0.0      # From standard classifier
    corrected_prevalence: float = 0.0  # After PU correction
    dark_figure_estimate: float = 0.0  # Estimated undetected positives

    # Diagnostics
    n_labelled_positive: int = 0
    n_unlabelled: int = 0
    mean_prob_on_positives: float = 0.0  # Should be high if classifier works

    def summary(self) -> str:
        return (
            f"PU Learning Results:\n"
            f"  Label frequency (c): {self.label_frequency:.3f}\n"
            f"  Interpretation: ~{self.label_frequency*100:.0f}% of true positives "
            f"are in the Retraction Watch data\n"
            f"  Raw prevalence: {self.raw_prevalence*100:.2f}%\n"
            f"  Corrected prevalence: {self.corrected_prevalence*100:.2f}%\n"
            f"  Dark figure: ~{self.dark_figure_estimate:.0f} undetected mill papers "
            f"in the corpus\n"
            f"  Labelled positives: {self.n_labelled_positive}, "
            f"Unlabelled: {self.n_unlabelled}"
        )


def estimate_label_frequency(
    y_true: np.ndarray,
    predicted_probs: np.ndarray,
    method: str = "e1",
) -> float:
    """Estimate the label frequency c = P(s=1 | y=1).

    This is the key quantity in PU learning: what fraction of true
    positives actually appear in our labelled set?

    Args:
        y_true: Binary labels (1 = labelled positive, 0 = unlabelled)
        predicted_probs: Classifier's predicted P(s=1 | x)
        method: Estimation method:
            "e1": Mean predicted probability on labelled positives (Elkan & Noto)
            "e2": Max predicted probability on labelled positives (conservative)
            "e3": Mean of top 50% of predicted probs on positives (robust)

    Returns:
        Estimated c, clipped to [0.01, 1.0]
    """
    positive_mask = y_true == 1
    if positive_mask.sum() == 0:
        logger.warning("No labelled positives — cannot estimate label frequency")
        return 1.0

    positive_probs = predicted_probs[positive_mask]

    if method == "e1":
        # Elkan & Noto estimator e1: average predicted prob on positives
        c = positive_probs.mean()
    elif method == "e2":
        # More conservative: maximum predicted prob on positives
        c = positive_probs.max()
    elif method == "e3":
        # Robust: mean of top 50% (avoids outlier positives that look negative)
        sorted_probs = np.sort(positive_probs)[::-1]
        top_half = sorted_probs[:max(len(sorted_probs) // 2, 1)]
        c = top_half.mean()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Clamp to reasonable range
    c = np.clip(c, 0.01, 1.0)

    logger.info(f"Estimated label frequency c={c:.3f} (method={method})")
    return float(c)


def estimate_label_frequency_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    n_folds: int = 5,
    method: str = "e1",
) -> float:
    """Estimate label frequency using cross-validation to avoid overfitting.

    Training and evaluating on the same data inflates c. This uses
    held-out predictions on positives for a less biased estimate.
    """
    n_pos = y.sum()
    if n_pos < 3:
        logger.warning(f"Too few positives ({n_pos}) for CV — using direct estimate")
        return 1.0

    cv = StratifiedKFold(
        n_splits=min(n_folds, n_pos),
        shuffle=True,
        random_state=42,
    )

    held_out_probs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        classifier.fit(X_train, y_train)
        probs = classifier.predict_proba(X_test)[:, 1]

        # Collect predictions on held-out positives
        pos_mask = y_test == 1
        if pos_mask.any():
            held_out_probs.extend(probs[pos_mask])

    if not held_out_probs:
        return 1.0

    held_out_probs = np.array(held_out_probs)

    if method == "e1":
        c = held_out_probs.mean()
    elif method == "e3":
        sorted_p = np.sort(held_out_probs)[::-1]
        top_half = sorted_p[:max(len(sorted_p) // 2, 1)]
        c = top_half.mean()
    else:
        c = held_out_probs.mean()

    c = np.clip(c, 0.01, 1.0)
    logger.info(f"CV-estimated label frequency c={c:.3f}")
    return float(c)


def correct_probabilities(
    raw_probs: np.ndarray,
    label_frequency: float,
) -> np.ndarray:
    """Apply PU correction to predicted probabilities.

    P(y=1 | x) = P(s=1 | x) / c

    where s=1 means "labelled positive" and y=1 means "truly positive".
    """
    corrected = raw_probs / label_frequency
    return np.clip(corrected, 0.0, 1.0)


def pu_classify(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    method: str = "e1",
    use_cv: bool = True,
    n_folds: int = 5,
) -> PUResult:
    """Full PU learning pipeline.

    Args:
        X: Feature matrix (scaled)
        y: Labels (1 = known positive/retracted, 0 = unlabelled)
        classifier: Sklearn-compatible classifier
        method: Label frequency estimation method
        use_cv: Use cross-validation for c estimation (recommended)
        n_folds: CV folds

    Returns:
        PUResult with corrected probabilities and prevalence
    """
    result = PUResult()
    result.n_labelled_positive = int(y.sum())
    result.n_unlabelled = int((y == 0).sum())

    # Step 1: Estimate label frequency
    if use_cv and result.n_labelled_positive >= 3:
        from sklearn.base import clone
        c = estimate_label_frequency_cv(
            X, y, clone(classifier), n_folds=n_folds, method=method
        )
    else:
        # Direct estimation (use with caution — overfits)
        classifier.fit(X, y)
        raw_probs = classifier.predict_proba(X)[:, 1]
        c = estimate_label_frequency(y, raw_probs, method=method)

    result.label_frequency = c
    result.mean_prob_on_positives = float(
        classifier.predict_proba(X)[:, 1][y == 1].mean()
        if hasattr(classifier, 'predict_proba') and y.sum() > 0
        else 0.0
    )

    # Step 2: Train final classifier on all data and get predictions
    classifier.fit(X, y)
    raw_probs = classifier.predict_proba(X)[:, 1]

    # Step 3: Correct probabilities
    corrected = correct_probabilities(raw_probs, c)
    result.corrected_probabilities = corrected

    # Step 4: Compute corrected prevalence
    result.raw_prevalence = float(raw_probs.mean())
    result.corrected_prevalence = float(corrected.mean())

    # Dark figure: estimated number of true positives minus labelled positives
    estimated_total_positives = corrected.sum()
    result.dark_figure_estimate = max(
        0, estimated_total_positives - result.n_labelled_positive
    )

    logger.info(result.summary())
    return result
