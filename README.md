# Paper Mill Subtypes in Medical AI — Analysis Code

## Citation

If you use this code, please cite:

> Farquhar, H. (2026). Analysis code for: Paper Mill Subtypes in Medical AI: Multi-Signal NLP Detection Reveals Heterogeneous Fraud Fingerprints. Zenodo. DOI: 10.5281/zenodo.19488868

**Associated pre-registration:** https://doi.org/10.17605/OSF.IO/JB4T6
**Pre-registered code (v1.0.0):** https://doi.org/10.5281/zenodo.19481250
**Preprint:** https://doi.org/10.31222/osf.io/3bvzc_v1

## Version History

- **v3.0.0** (2026-04-24): Post-revision release. Adds leave-Hindawi-out sensitivity analysis script. No changes to pre-registered classifier, weights, or features. See CHANGELOG.md.
- **v2.0.0** (2026-04-09): Post-analysis release with bug fixes.
- **v1.0.0** (2026-04-09): Pre-registered code deposited before any analyses were run.

## Overview

This repository contains the complete analysis pipeline for detecting paper mill characteristics in the medical artificial intelligence (AI) literature. The pipeline collects paper metadata from OpenAlex, extracts multi-signal detection features, trains a supervised classifier using Retraction Watch labels, applies Positive-Unlabelled (PU) learning correction, and estimates the prevalence of paper mill characteristics with confidence intervals.

The analysis plan was pre-registered on the Open Science Framework (OSF) on 9 April 2026, prior to classifier training or any inferential analysis.

## Code Structure

```
src/
├── collect/                    # Data collection
│   ├── openalex_collector.py   # Retrieve medical AI papers from OpenAlex
│   ├── pmc_fetcher.py          # Download full-text XML from PubMed Central
│   └── retraction_loader.py    # Load/filter Retraction Watch labels
│
├── detect/                     # Detection features (7 categories)
│   ├── tortured_phrases.py     # Synonym-substitution detection (94 phrases)
│   ├── structure_scorer.py     # Formulaic structure and boilerplate
│   ├── ai_text_detector.py     # LLM-generated text markers
│   ├── citation_analyser.py    # Citation pattern anomalies
│   ├── similarity_detector.py  # Cross-document and methods-section similarity
│   ├── author_network.py       # Co-authorship network features
│   └── bibliometric_flags.py   # Author/journal metadata anomalies
│
├── classify/                   # Classification and validation
│   ├── feature_builder.py      # Unified feature matrix construction
│   ├── classifier.py           # RF + XGBoost ensemble classifier
│   ├── pu_learning.py          # Positive-Unlabelled learning (Elkan & Noto 2008)
│   └── validator.py            # Retraction Watch validation + prevalence CIs
│
├── analyse/                    # Downstream analyses
│   ├── prevalence.py           # Overall and subgroup prevalence estimation
│   ├── temporal_trends.py      # Pre/post-ChatGPT temporal analysis
│   ├── journal_analysis.py     # Journal-level prevalence comparison
│   └── geographic.py           # WHO regional analysis + sensitivity test
│
└── pipeline.py                 # Pipeline orchestrator + unsupervised scoring
```

## Requirements

- Python >= 3.10
- Dependencies listed in `pyproject.toml`:
  - pyalex >= 0.14
  - requests >= 2.31
  - lxml >= 5.0
  - pandas >= 2.0
  - scikit-learn >= 1.4
  - xgboost >= 2.0
  - numpy >= 1.26
  - tqdm >= 4.66
  - matplotlib >= 3.8
  - seaborn >= 0.13

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Run the full pipeline (small test):

```bash
python src/pipeline.py
```

Or run individual stages:

```python
from src.pipeline import stage_collect, stage_detect, stage_classify, stage_analyse

corpus_df = stage_collect(max_papers=1000)
feature_df = stage_detect(corpus_df)
predictions_df = stage_classify(feature_df, retraction_path="data/retraction_watch/retraction_watch.csv", corpus_df=corpus_df)
stage_analyse(predictions_df)
```

## Data Sources

All data sources are freely available:

| Dataset | URL | Licence |
|---------|-----|---------|
| OpenAlex | https://openalex.org | CC0 |
| PMC Open Access | https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/ | Public domain |
| Retraction Watch | https://api.labs.crossref.org/data/retractionwatch | Free via Crossref |

## Pre-Registration

The analysis plan for this study was pre-registered on OSF prior to classifier training or prevalence estimation:

- **DOI:** https://doi.org/10.17605/OSF.IO/JB4T6
- **OSF Project:** https://osf.io/njh4e
- **Type:** Post-data, pre-analysis
- **Date:** 9 April 2026

## Licence

MIT License

Copyright (c) 2026 Hayden Farquhar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

Hayden Farquhar
Independent Researcher, Finley, NSW, Australia
