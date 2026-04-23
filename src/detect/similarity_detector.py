"""
Cross-document similarity detection for paper mill batch identification.

Paper mills produce batches of structurally similar papers. This module
computes pairwise similarity features that capture this batch signal:
- TF-IDF cosine similarity of abstracts
- Maximum similarity to any other paper in the corpus (nearest-neighbour)
- Count of near-duplicates above a threshold
- Cluster membership in high-similarity groups

These are among the strongest signals for paper mill detection because
legitimate independent research rarely produces near-identical text.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Similarity thresholds
HIGH_SIMILARITY_THRESHOLD = 0.85  # Near-duplicate
MODERATE_SIMILARITY_THRESHOLD = 0.65  # Suspiciously similar


def compute_corpus_similarity(
    texts: list[str],
    ids: Optional[list[str]] = None,
    max_features: int = 5000,
) -> dict:
    """Compute pairwise TF-IDF similarity across the corpus.

    Args:
        texts: List of document texts (abstracts or body text)
        ids: Optional document identifiers
        max_features: Maximum TF-IDF vocabulary size

    Returns:
        Dict with similarity matrix and per-document features
    """
    if not texts or len(texts) < 2:
        return {"features": [], "similarity_matrix": None}

    # Filter out empty texts
    valid_mask = [bool(t and len(t.strip()) > 50) for t in texts]
    valid_texts = [t for t, v in zip(texts, valid_mask) if v]
    valid_ids = (
        [i for i, v in zip(ids, valid_mask) if v]
        if ids else list(range(len(valid_texts)))
    )

    if len(valid_texts) < 2:
        return {"features": [], "similarity_matrix": None}

    logger.info(f"Computing TF-IDF similarity for {len(valid_texts)} documents")

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),  # Unigrams + bigrams for better phrase matching
        min_df=2,  # Must appear in at least 2 documents
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(valid_texts)

    # Pairwise cosine similarity
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Zero out diagonal (self-similarity = 1.0, not informative)
    np.fill_diagonal(sim_matrix, 0.0)

    # Per-document features
    features = []
    for i in range(len(valid_texts)):
        row = sim_matrix[i]

        max_sim = row.max()
        mean_sim = row.mean()
        n_high = int((row >= HIGH_SIMILARITY_THRESHOLD).sum())
        n_moderate = int((row >= MODERATE_SIMILARITY_THRESHOLD).sum())

        # Nearest neighbour
        nn_idx = row.argmax()

        features.append({
            "id": valid_ids[i],
            "sim_max": round(float(max_sim), 4),
            "sim_mean": round(float(mean_sim), 4),
            "sim_n_high": n_high,
            "sim_n_moderate": n_moderate,
            "sim_nearest_id": valid_ids[nn_idx],
            "sim_nearest_score": round(float(max_sim), 4),
        })

    logger.info(
        f"Similarity stats: "
        f"mean max_sim={np.mean([f['sim_max'] for f in features]):.3f}, "
        f"papers with high-similarity match="
        f"{sum(1 for f in features if f['sim_n_high'] > 0)}"
    )

    return {
        "features": features,
        "similarity_matrix": sim_matrix,
        "ids": valid_ids,
    }


def find_similar_clusters(
    similarity_result: dict,
    threshold: float = HIGH_SIMILARITY_THRESHOLD,
    min_cluster_size: int = 3,
) -> list[list[str]]:
    """Find clusters of highly similar papers (potential mill batches).

    Uses connected-component clustering on the similarity graph:
    papers linked if similarity >= threshold.

    Args:
        similarity_result: Output from compute_corpus_similarity
        threshold: Minimum similarity to form an edge
        min_cluster_size: Minimum papers to report a cluster

    Returns:
        List of clusters (each cluster is a list of paper IDs)
    """
    sim_matrix = similarity_result.get("similarity_matrix")
    ids = similarity_result.get("ids", [])

    if sim_matrix is None or len(ids) < min_cluster_size:
        return []

    n = len(ids)

    # Build adjacency from threshold
    adj = sim_matrix >= threshold

    # Connected components via BFS
    visited = set()
    clusters = []

    for start in range(n):
        if start in visited:
            continue

        # BFS
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in range(n):
                if neighbor not in visited and adj[node, neighbor]:
                    queue.append(neighbor)

        if len(component) >= min_cluster_size:
            clusters.append([ids[i] for i in component])

    logger.info(
        f"Found {len(clusters)} suspicious clusters "
        f"(threshold={threshold}, min_size={min_cluster_size})"
    )

    return clusters


def add_similarity_features(
    feature_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    text_column: str = "abstract",
    id_column: str = "openalex_id",
) -> pd.DataFrame:
    """Add similarity features to the feature DataFrame.

    Args:
        feature_df: Existing feature DataFrame
        corpus_df: Corpus with text column
        text_column: Column containing text for similarity
        id_column: Document ID column

    Returns:
        feature_df with similarity columns added
    """
    texts = corpus_df[text_column].fillna("").tolist()
    ids = corpus_df[id_column].tolist()

    result = compute_corpus_similarity(texts, ids)

    if not result["features"]:
        feature_df["sim_max"] = 0.0
        feature_df["sim_mean"] = 0.0
        feature_df["sim_n_high"] = 0
        feature_df["sim_n_moderate"] = 0
        return feature_df

    sim_df = pd.DataFrame(result["features"])
    sim_df = sim_df.rename(columns={"id": id_column})

    feature_df = feature_df.merge(
        sim_df[[id_column, "sim_max", "sim_mean", "sim_n_high", "sim_n_moderate"]],
        on=id_column,
        how="left",
    )

    # Fill papers that weren't in similarity analysis (e.g., empty abstracts)
    for col in ["sim_max", "sim_mean", "sim_n_high", "sim_n_moderate"]:
        feature_df[col] = feature_df[col].fillna(0.0)

    return feature_df


# ---------------------------------------------------------------------------
# Methods-section similarity (full-text only)
# ---------------------------------------------------------------------------

def extract_methods_text(parsed_article: dict) -> str:
    """Extract methods section text from a parsed PMC article.

    Looks for sections with methods-related headings. Falls back to
    the second quarter of body text if no explicit methods heading found
    (many papers have methods in that position even without a heading).
    """
    METHODS_KEYWORDS = {
        "methods", "methodology", "materials and methods",
        "experimental", "study design", "statistical analysis",
        "data collection", "participants", "procedures",
    }

    sections = parsed_article.get("sections", [])

    # Try explicit heading match first
    methods_parts = []
    for sec in sections:
        heading = sec.get("heading", "").strip().lower()
        # Remove numbering
        import re
        heading_clean = re.sub(r"^\d+[\.\)]\s*", "", heading)
        if any(kw in heading_clean for kw in METHODS_KEYWORDS):
            methods_parts.append(sec.get("text", ""))

    if methods_parts:
        return " ".join(methods_parts)

    # Fallback: use second quarter of body text (heuristic for IMRAD position)
    body = parsed_article.get("body_text", "")
    if len(body) > 200:
        quarter = len(body) // 4
        return body[quarter:quarter * 2]

    return ""


def compute_methods_similarity(
    fulltext_data: list[dict],
    max_features: int = 3000,
) -> dict:
    """Compute pairwise similarity on methods sections only.

    Methods sections are the highest-value text for mill detection because
    mills reuse them near-verbatim, whereas abstracts may be more varied.

    Args:
        fulltext_data: List of parsed PMC articles (from pmc_fetcher)
        max_features: TF-IDF vocabulary size

    Returns:
        Dict with per-document features keyed by pmcid
    """
    # Extract methods text from each article
    valid_articles = []
    for article in fulltext_data:
        methods_text = extract_methods_text(article)
        if len(methods_text) > 100:  # Need substantial text
            valid_articles.append({
                "pmcid": article.get("pmcid", ""),
                "methods_text": methods_text,
            })

    if len(valid_articles) < 2:
        logger.info("Too few articles with methods sections for similarity analysis")
        return {"features": {}}

    texts = [a["methods_text"] for a in valid_articles]
    ids = [a["pmcid"] for a in valid_articles]

    logger.info(f"Computing methods-section similarity for {len(texts)} articles")

    result = compute_corpus_similarity(texts, ids, max_features=max_features)

    # Convert to dict keyed by pmcid for easy merge
    features_by_id = {}
    for feat in result.get("features", []):
        pmcid = feat["id"]
        features_by_id[pmcid] = {
            "methods_sim_max": feat["sim_max"],
            "methods_sim_mean": feat["sim_mean"],
            "methods_sim_n_high": feat["sim_n_high"],
            "methods_sim_n_moderate": feat["sim_n_moderate"],
            "methods_sim_nearest_id": feat.get("sim_nearest_id", ""),
            "methods_sim_nearest_score": feat.get("sim_nearest_score", 0.0),
        }

    # Also find methods-specific clusters
    clusters = find_similar_clusters(
        result,
        threshold=HIGH_SIMILARITY_THRESHOLD,
        min_cluster_size=2,  # Even pairs are suspicious for methods
    )

    # Mark papers that are in a methods-similarity cluster
    papers_in_cluster = set()
    for cluster in clusters:
        papers_in_cluster.update(cluster)

    for pmcid, feat in features_by_id.items():
        feat["methods_in_cluster"] = pmcid in papers_in_cluster

    logger.info(
        f"Methods similarity: {len(features_by_id)} articles analysed, "
        f"{len(clusters)} suspicious clusters found, "
        f"{len(papers_in_cluster)} papers in clusters"
    )

    return {"features": features_by_id, "clusters": clusters}


def add_methods_similarity_features(
    feature_df: pd.DataFrame,
    fulltext_data: list[dict],
    id_column: str = "pmcid",
) -> pd.DataFrame:
    """Add methods-section similarity features to the feature DataFrame.

    Only papers with PMC full text will have these features populated;
    others get 0.0 (conservative — no evidence of similarity).
    """
    result = compute_methods_similarity(fulltext_data)
    features_by_id = result.get("features", {})

    if not features_by_id:
        for col in ["methods_sim_max", "methods_sim_mean",
                     "methods_sim_n_high", "methods_in_cluster"]:
            feature_df[col] = 0.0
        return feature_df

    # Map features to DataFrame
    new_cols = {
        "methods_sim_max": [],
        "methods_sim_mean": [],
        "methods_sim_n_high": [],
        "methods_in_cluster": [],
    }

    for _, row in feature_df.iterrows():
        pmcid = row.get(id_column, "")
        feat = features_by_id.get(pmcid, {})
        new_cols["methods_sim_max"].append(feat.get("methods_sim_max", 0.0))
        new_cols["methods_sim_mean"].append(feat.get("methods_sim_mean", 0.0))
        new_cols["methods_sim_n_high"].append(feat.get("methods_sim_n_high", 0))
        new_cols["methods_in_cluster"].append(
            1.0 if feat.get("methods_in_cluster", False) else 0.0
        )

    for col, values in new_cols.items():
        feature_df[col] = values

    n_with_methods = sum(
        1 for _, row in feature_df.iterrows()
        if row.get("methods_sim_max", 0) > 0
    )
    logger.info(f"Methods similarity features added for {n_with_methods} papers")

    return feature_df
