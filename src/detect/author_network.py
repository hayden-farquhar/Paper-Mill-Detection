"""
Author co-authorship network features for paper mill detection.

Paper mills maintain "author pools" — sets of names reused across
manufactured papers. This creates distinctive network signatures:
- Small, dense cliques of co-authors
- Authors with unusually high co-authorship overlap
- Rapid author-pair formation (new collaborators appearing on many papers quickly)

This module builds a co-authorship graph from the corpus and extracts
per-paper network features without requiring external graph libraries
(uses only numpy/pandas for Colab CPU compatibility).
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AuthorNetworkFeatures:
    """Network-based features for a single paper."""

    # Author reuse: how many of this paper's authors appear on other
    # corpus papers? High values suggest an author pool.
    author_reuse_rate: float = 0.0  # fraction of authors on >1 corpus paper
    max_author_corpus_papers: int = 0  # most prolific author's corpus paper count

    # Co-authorship density: among this paper's authors, what fraction
    # of all possible pairs have co-authored other papers in the corpus?
    coauthor_density: float = 0.0  # 0-1, fraction of author-pairs with prior ties

    # Shared co-author fraction: for each author on this paper, what
    # fraction of their other corpus co-authors also appear on this paper?
    # High values = insular group that always publishes together.
    shared_coauthor_fraction: float = 0.0

    # Clique score: is this paper's author set a subset of a known
    # dense clique in the co-authorship graph?
    clique_score: float = 0.0

    # Author pair novelty: what fraction of co-author pairs on this paper
    # have never co-authored before in the corpus?
    # Very low novelty = the same group keeps publishing together.
    pair_novelty: float = 1.0

    def to_dict(self) -> dict:
        return {
            "net_author_reuse_rate": round(self.author_reuse_rate, 4),
            "net_max_author_papers": self.max_author_corpus_papers,
            "net_coauthor_density": round(self.coauthor_density, 4),
            "net_shared_coauthor_fraction": round(self.shared_coauthor_fraction, 4),
            "net_clique_score": round(self.clique_score, 4),
            "net_pair_novelty": round(self.pair_novelty, 4),
        }


class AuthorNetworkAnalyser:
    """Build and analyse the co-authorship network from a paper corpus."""

    def __init__(self):
        # author_id -> set of paper_ids they appear on
        self.author_papers: dict[str, set[str]] = defaultdict(set)
        # author_id -> set of co-author_ids (across all their papers)
        self.author_coauthors: dict[str, set[str]] = defaultdict(set)
        # frozenset(author_id, author_id) -> count of papers they co-authored
        self.pair_counts: Counter = Counter()
        # paper_id -> list of author_ids
        self.paper_authors: dict[str, list[str]] = {}

    def build_network(self, corpus_df: pd.DataFrame):
        """Build the co-authorship network from corpus metadata.

        Expects corpus_df to have:
        - openalex_id: paper identifier
        - authorships data (we extract from the raw OpenAlex fields)

        For the initial build, we use the collected metadata which includes
        first_author_id. For richer network analysis, we need full authorship
        lists from OpenAlex, stored as a semicolon-separated field.
        """
        logger.info(f"Building co-authorship network from {len(corpus_df)} papers")

        for _, row in corpus_df.iterrows():
            paper_id = row.get("openalex_id", "")
            if not paper_id:
                continue

            # Extract author IDs — try multiple possible column formats
            author_ids = self._extract_author_ids(row)
            if not author_ids:
                continue

            self.paper_authors[paper_id] = author_ids

            # Register each author
            for aid in author_ids:
                self.author_papers[aid].add(paper_id)

            # Register all co-author pairs
            for i in range(len(author_ids)):
                for j in range(i + 1, len(author_ids)):
                    pair = frozenset([author_ids[i], author_ids[j]])
                    self.pair_counts[pair] += 1
                    self.author_coauthors[author_ids[i]].add(author_ids[j])
                    self.author_coauthors[author_ids[j]].add(author_ids[i])

        logger.info(
            f"Network: {len(self.author_papers)} authors, "
            f"{len(self.pair_counts)} co-author pairs, "
            f"{len(self.paper_authors)} papers with author data"
        )

    def _extract_author_ids(self, row: pd.Series) -> list[str]:
        """Extract author IDs from a corpus row.

        Handles multiple possible column formats.
        """
        # Try 'author_ids' column (semicolon-separated OpenAlex IDs)
        if "author_ids" in row.index:
            ids_str = str(row.get("author_ids", ""))
            if ids_str and ids_str != "nan":
                return [a.strip() for a in ids_str.split(";") if a.strip()]

        # Fall back to first_author_id only (less information, but still useful
        # for the first-author reuse signal)
        first_id = row.get("first_author_id", "")
        if first_id and str(first_id) != "nan":
            return [str(first_id).strip()]

        return []

    def compute_features(self, paper_id: str) -> AuthorNetworkFeatures:
        """Compute network features for a single paper."""
        features = AuthorNetworkFeatures()

        author_ids = self.paper_authors.get(paper_id, [])
        if not author_ids:
            return features

        n_authors = len(author_ids)

        # --- Author reuse rate ---
        # What fraction of this paper's authors appear on other corpus papers?
        reuse_count = 0
        max_papers = 0
        for aid in author_ids:
            n_papers = len(self.author_papers.get(aid, set()))
            if n_papers > 1:
                reuse_count += 1
            max_papers = max(max_papers, n_papers)

        features.author_reuse_rate = reuse_count / n_authors
        features.max_author_corpus_papers = max_papers

        if n_authors < 2:
            return features

        # --- Co-author density ---
        # Among this paper's authors, what fraction of all possible pairs
        # have co-authored on OTHER papers in the corpus?
        n_possible_pairs = n_authors * (n_authors - 1) // 2
        external_pairs = 0

        for i in range(n_authors):
            for j in range(i + 1, n_authors):
                pair = frozenset([author_ids[i], author_ids[j]])
                # pair_counts includes this paper, so >1 means they co-authored elsewhere too
                if self.pair_counts.get(pair, 0) > 1:
                    external_pairs += 1

        features.coauthor_density = external_pairs / n_possible_pairs

        # --- Shared co-author fraction ---
        # For each author, what fraction of their corpus co-authors are also on this paper?
        paper_author_set = set(author_ids)
        shared_fractions = []
        for aid in author_ids:
            all_coauthors = self.author_coauthors.get(aid, set())
            if len(all_coauthors) > 0:
                # Exclude self
                other_coauthors = all_coauthors - {aid}
                if other_coauthors:
                    overlap = len(paper_author_set & other_coauthors)
                    shared_fractions.append(overlap / len(other_coauthors))

        if shared_fractions:
            features.shared_coauthor_fraction = float(np.mean(shared_fractions))

        # --- Clique score ---
        # How close is this author set to a complete subgraph?
        # (coauthor_density already captures part of this; clique_score
        # also weights by the strength of pairwise connections)
        if n_possible_pairs > 0:
            pair_strengths = []
            for i in range(n_authors):
                for j in range(i + 1, n_authors):
                    pair = frozenset([author_ids[i], author_ids[j]])
                    count = self.pair_counts.get(pair, 0)
                    # Normalise: >3 co-authored papers is strong evidence
                    pair_strengths.append(min(count / 3.0, 1.0))

            features.clique_score = float(np.mean(pair_strengths))

        # --- Pair novelty ---
        # What fraction of co-author pairs appear ONLY on this paper?
        novel_pairs = 0
        for i in range(n_authors):
            for j in range(i + 1, n_authors):
                pair = frozenset([author_ids[i], author_ids[j]])
                if self.pair_counts.get(pair, 0) <= 1:
                    novel_pairs += 1

        features.pair_novelty = novel_pairs / n_possible_pairs if n_possible_pairs > 0 else 1.0

        return features

    def compute_features_batch(self, corpus_df: pd.DataFrame) -> pd.DataFrame:
        """Compute network features for all papers in the corpus.

        Returns DataFrame with network features keyed by openalex_id.
        """
        records = []
        for _, row in corpus_df.iterrows():
            paper_id = row.get("openalex_id", "")
            features = self.compute_features(paper_id)
            record = features.to_dict()
            record["openalex_id"] = paper_id
            records.append(record)

        df = pd.DataFrame(records)

        # Log summary statistics
        if len(df) > 0:
            logger.info(
                f"Author network features: "
                f"mean reuse_rate={df['net_author_reuse_rate'].mean():.3f}, "
                f"mean coauthor_density={df['net_coauthor_density'].mean():.3f}, "
                f"papers with high reuse (>0.5)="
                f"{(df['net_author_reuse_rate'] > 0.5).sum()}"
            )

        return df

    def find_author_pools(
        self,
        min_shared_papers: int = 3,
        min_pool_size: int = 3,
    ) -> list[dict]:
        """Identify potential author pools — dense cliques of co-authors.

        An author pool is a group of authors who co-author with each other
        on multiple corpus papers at rates suggesting coordinated production
        rather than genuine collaboration.

        Returns list of pool dicts with members, paper counts, and density.
        """
        # Find authors who frequently co-author together
        # Start from the strongest pairs and grow cliques
        strong_pairs = [
            (pair, count) for pair, count in self.pair_counts.items()
            if count >= min_shared_papers
        ]

        if not strong_pairs:
            return []

        # Build adjacency among authors with strong ties
        strong_adj: dict[str, set[str]] = defaultdict(set)
        for pair, count in strong_pairs:
            a, b = list(pair)
            strong_adj[a].add(b)
            strong_adj[b].add(a)

        # Find connected components in the strong-tie graph
        visited = set()
        pools = []

        for start_author in strong_adj:
            if start_author in visited:
                continue

            component = set()
            queue = [start_author]
            while queue:
                author = queue.pop(0)
                if author in visited:
                    continue
                visited.add(author)
                component.add(author)
                for neighbour in strong_adj[author]:
                    if neighbour not in visited:
                        queue.append(neighbour)

            if len(component) >= min_pool_size:
                # Compute pool statistics
                pool_papers = set()
                for aid in component:
                    pool_papers.update(self.author_papers.get(aid, set()))

                # Internal density: what fraction of possible pairs are connected?
                members = list(component)
                n = len(members)
                connected_pairs = 0
                total_pairs = n * (n - 1) // 2
                for i in range(n):
                    for j in range(i + 1, n):
                        pair = frozenset([members[i], members[j]])
                        if self.pair_counts.get(pair, 0) >= min_shared_papers:
                            connected_pairs += 1

                density = connected_pairs / total_pairs if total_pairs > 0 else 0

                pools.append({
                    "members": list(component),
                    "size": len(component),
                    "papers": list(pool_papers),
                    "n_papers": len(pool_papers),
                    "density": round(density, 3),
                })

        pools.sort(key=lambda p: p["n_papers"], reverse=True)
        logger.info(f"Found {len(pools)} potential author pools")

        return pools
