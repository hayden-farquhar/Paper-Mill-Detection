"""
Detect tortured phrases in academic text.

Tortured phrases are nonsensical synonym substitutions used by paper mills
to evade plagiarism detection. For example:
  - "deep learning" -> "profound learning"
  - "artificial intelligence" -> "counterfeit consciousness"
  - "random forest" -> "arbitrary woodland"
  - "breast cancer" -> "bosom malignant growth"
  - "support vector machine" -> "backing point machine"

We use a curated dictionary from the Problematic Paper Screener project
plus domain-specific additions for medical AI.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Built-in dictionary of tortured phrases commonly found in medical AI papers.
# Format: tortured_phrase -> correct_phrase
# Source: Compiled from Problematic Paper Screener + Cabanac et al. (2021)
# and extended with medical AI-specific entries.
TORTURED_PHRASES = {
    # AI/ML terminology
    "profound learning": "deep learning",
    "profound brain network": "deep neural network",
    "fake brain network": "artificial neural network",
    "counterfeit brain network": "artificial neural network",
    "counterfeit neural network": "artificial neural network",
    "counterfeit intelligence": "artificial intelligence",
    "fake intelligence": "artificial intelligence",
    "sham intelligence": "artificial intelligence",
    "arbitrary forest": "random forest",
    "arbitrary woodland": "random forest",
    "irregular woodland": "random forest",
    "stochastic forest": "random forest",
    "backing point machine": "support vector machine",
    "backing vector machine": "support vector machine",
    "support vector apparatus": "support vector machine",
    "recurrent brain network": "recurrent neural network",
    "long haul memory": "long short-term memory",
    "long momentary memory": "long short-term memory",
    "convolutional brain network": "convolutional neural network",
    "convolution brain network": "convolutional neural network",
    "generative ill-disposed network": "generative adversarial network",
    "generative antagonistic network": "generative adversarial network",
    "element extraction": "feature extraction",
    "highlight extraction": "feature extraction",
    "characteristic extraction": "feature extraction",
    "include extraction": "feature extraction",
    "choice tree": "decision tree",
    "choice woodland": "decision tree",
    "calculated inclination descent": "gradient descent",
    "angle descent": "gradient descent",
    "back proliferation": "backpropagation",
    "back engendering": "backpropagation",
    "precision and review": "precision and recall",
    "exactness and review": "precision and recall",
    "exchange learning": "transfer learning",
    "move learning": "transfer learning",
    "semi-supervised gaining knowledge": "semi-supervised learning",
    "unaided learning": "unsupervised learning",
    # "directed learning" removed — ambiguous; used in education (pedagogy) contexts
    "managed learning": "supervised learning",
    "managed gaining knowledge": "supervised learning",
    # Medical terminology
    "bosom malignant growth": "breast cancer",
    "chest malignant growth": "breast cancer",
    "bosom disease": "breast cancer",
    "prostate malignant growth": "prostate cancer",
    "lung malignant growth": "lung cancer",
    "colorectal malignant growth": "colorectal cancer",
    "cerebrum tumor": "brain tumor",
    "cerebrum tumour": "brain tumour",
    "sham treatment": "placebo",
    "sham examination": "placebo test",
    "control gathering": "control group",
    "benchmark bunch": "control group",
    "attractive reverberation imaging": "magnetic resonance imaging",
    "attractive reverberance imaging": "magnetic resonance imaging",
    "processed tomography": "computed tomography",
    "figured tomography": "computed tomography",
    "x-beam": "X-ray",
    "bright imaging": "fluorescence imaging",
    "diabetes mellitus sort 2": "diabetes mellitus type 2",
    "coronary illness": "heart disease",
    "cardiovascular breakdown": "heart failure",
    "pneumonic embolism": "pulmonary embolism",
    # Statistical terminology
    "region under the bend": "area under the curve",
    "zone under the bend": "area under the curve",
    "collector working trademark": "receiver operating characteristic",
    "beneficiary working trademark": "receiver operating characteristic",
    "collector working trademark bend": "ROC curve",
    "affirmation coefficient": "correlation coefficient",
    "connection coefficient": "correlation coefficient",
    "relapse investigation": "regression analysis",
    "relapse examination": "regression analysis",
    "straight relapse": "linear regression",
    "calculated relapse": "logistic regression",
    "strategic relapse": "logistic regression",
    "strategic regression": "logistic regression",
    "chi-square examination": "chi-square test",
    "standard deviation esteem": "standard deviation value",
    "measurable importance": "statistical significance",
    "factual importance": "statistical significance",
    "factual significance": "statistical significance",
    "invalid speculation": "null hypothesis",
    "void theory": "null hypothesis",
    # Publication/methods terminology
    "writing audit": "literature review",
    "writing survey": "literature review",
    "orderly survey": "systematic review",
    "efficient survey": "systematic review",
    "meta examination": "meta-analysis",
    "meta-investigation": "meta-analysis",
    "cross-sectional examination": "cross-sectional study",
    "partner study": "cohort study",
    "forthcoming investigation": "prospective study",
    # "review study" removed — ambiguous; legitimate in education/methods contexts
    "educated assent": "informed consent",
    "moral endorsement": "ethical approval",
    "moral board": "ethics board",
}


@dataclass
class TorturedPhraseMatch:
    """A single tortured phrase match in text."""
    tortured: str
    correct: str
    start: int
    end: int
    context: str  # surrounding text for inspection


@dataclass
class TorturedPhraseResult:
    """Results of tortured phrase detection on a single document."""
    matches: list[TorturedPhraseMatch] = field(default_factory=list)
    word_count: int = 0

    @property
    def count(self) -> int:
        return len(self.matches)

    @property
    def density(self) -> float:
        """Tortured phrases per 1,000 words."""
        if self.word_count == 0:
            return 0.0
        return (self.count / self.word_count) * 1000

    @property
    def unique_phrases(self) -> int:
        """Number of distinct tortured phrases found."""
        return len(set(m.tortured for m in self.matches))

    def to_dict(self) -> dict:
        return {
            "tortured_phrase_count": self.count,
            "tortured_phrase_density": round(self.density, 4),
            "tortured_unique_count": self.unique_phrases,
            "word_count": self.word_count,
            "matches": [
                {"tortured": m.tortured, "correct": m.correct, "context": m.context}
                for m in self.matches
            ],
        }


class TorturedPhraseDetector:
    """Detect tortured phrases in text using dictionary matching."""

    def __init__(
        self,
        dictionary: Optional[dict[str, str]] = None,
        extra_phrases_path: Optional[Path] = None,
    ):
        """Initialize detector with tortured phrase dictionary.

        Args:
            dictionary: Custom {tortured: correct} mapping. If None, uses built-in.
            extra_phrases_path: Path to JSON file with additional phrases.
        """
        self.phrases = dict(TORTURED_PHRASES)

        if dictionary:
            self.phrases.update(dictionary)

        if extra_phrases_path and Path(extra_phrases_path).exists():
            with open(extra_phrases_path) as f:
                extra = json.load(f)
            self.phrases.update(extra)
            logger.info(f"Loaded {len(extra)} extra phrases from {extra_phrases_path}")

        # Compile regex patterns for efficient matching
        # Sort by length (longest first) to match longer phrases first
        sorted_phrases = sorted(self.phrases.keys(), key=len, reverse=True)
        self._patterns = [
            (phrase, re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE))
            for phrase in sorted_phrases
        ]

        logger.info(f"Initialized detector with {len(self._patterns)} phrases")

    def detect(self, text: str, context_window: int = 50) -> TorturedPhraseResult:
        """Detect tortured phrases in text.

        Args:
            text: The text to analyse
            context_window: Number of characters of context to include around each match

        Returns:
            TorturedPhraseResult with all matches and statistics
        """
        if not text:
            return TorturedPhraseResult()

        word_count = len(text.split())
        matches = []

        for phrase, pattern in self._patterns:
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()

                # Extract surrounding context
                ctx_start = max(0, start - context_window)
                ctx_end = min(len(text), end + context_window)
                context = text[ctx_start:ctx_end].replace("\n", " ").strip()

                matches.append(TorturedPhraseMatch(
                    tortured=match.group(),
                    correct=self.phrases[phrase],
                    start=start,
                    end=end,
                    context=f"...{context}...",
                ))

        result = TorturedPhraseResult(matches=matches, word_count=word_count)
        return result

    def detect_batch(
        self,
        texts: list[str],
        ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Detect tortured phrases in multiple texts.

        Args:
            texts: List of text strings to analyse
            ids: Optional list of document IDs (parallel to texts)

        Returns:
            List of result dicts with detection features
        """
        results = []
        for i, text in enumerate(texts):
            result = self.detect(text or "")
            record = result.to_dict()
            if ids:
                record["id"] = ids[i]
            results.append(record)

        return results


def load_problematic_paper_screener_dict(filepath: Path) -> dict[str, str]:
    """Load tortured phrase dictionary from Problematic Paper Screener export.

    The PPS provides phrase pairs in various formats. This loader handles
    CSV format with columns: tortured_phrase, correct_phrase.
    """
    import csv

    phrases = {}
    filepath = Path(filepath)

    if filepath.suffix == ".json":
        with open(filepath) as f:
            phrases = json.load(f)
    elif filepath.suffix == ".csv":
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tortured = row.get("tortured_phrase", row.get("tortured", "")).strip()
                correct = row.get("correct_phrase", row.get("correct", "")).strip()
                if tortured and correct:
                    phrases[tortured.lower()] = correct.lower()

    logger.info(f"Loaded {len(phrases)} phrases from {filepath}")
    return phrases


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = TorturedPhraseDetector()

    # Test with sample text containing tortured phrases
    test_text = """
    In this study, we propose a profound learning approach for bosom malignant
    growth detection using attractive reverberation imaging. Our counterfeit
    brain network architecture combines element extraction with an arbitrary
    forest classifier. The region under the bend achieved 0.95, demonstrating
    factual importance (p < 0.001). A writing survey of related methods was
    conducted following moral endorsement from the ethics committee. The
    calculated relapse model was used as a baseline for comparison.
    """

    result = detector.detect(test_text)
    print(f"\nDetected {result.count} tortured phrases ({result.density:.1f} per 1000 words)")
    print(f"Unique phrases: {result.unique_phrases}")
    print()
    for m in result.matches:
        print(f'  "{m.tortured}" -> should be "{m.correct}"')
        print(f"    Context: {m.context[:100]}")
        print()
