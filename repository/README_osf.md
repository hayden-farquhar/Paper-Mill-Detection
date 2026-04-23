# Paper Mill Detection Pipeline — Pre-Registered Analysis Code

**OSF Project:** https://osf.io/njh4e
**Pre-Registration DOI:** https://doi.org/10.17605/OSF.IO/JB4T6

## Purpose

This code implements the analysis pipeline described in the pre-registration document. It was written prior to classifier training, prevalence estimation, or any inferential analysis. Uploading it to the OSF project provides verifiable evidence that the implementation matches the pre-registered plan.

## Code Structure

```
src/
├── collect/                    # Data collection (Section 3 of pre-registration)
│   ├── openalex_collector.py   # Retrieves medical AI papers from OpenAlex
│   ├── pmc_fetcher.py          # Downloads full-text XML from PMC
│   └── retraction_loader.py    # Loads Retraction Watch data as positive labels
│
├── detect/                     # Detection features (Section 4.2)
│   ├── tortured_phrases.py     # Category 1: Tortured phrase detection (94 phrases)
│   ├── structure_scorer.py     # Category 2: Structural formulaicity
│   ├── ai_text_detector.py     # Category 3: AI-generated text markers
│   ├── citation_analyser.py    # Category 4: Citation anomalies
│   ├── similarity_detector.py  # Category 5: Cross-document + methods similarity
│   ├── author_network.py       # Category 6: Co-authorship network features
│   └── bibliometric_flags.py   # Category 7 (partial): Bibliometric metadata
│
├── classify/                   # Classification (Section 5)
│   ├── feature_builder.py      # Combines all features into unified matrix
│   ├── classifier.py           # RF + XGBoost ensemble (Section 5.2)
│   ├── pu_learning.py          # Positive-Unlabelled correction (Section 5.5)
│   └── validator.py            # Retraction Watch validation + prevalence CIs
│
├── analyse/                    # Analyses (Sections 5.7–5.9)
│   ├── prevalence.py           # Overall and subgroup prevalence
│   ├── temporal_trends.py      # Pre/post-ChatGPT comparison (Section 5.7)
│   ├── journal_analysis.py     # Journal-level analysis (Section 5.9)
│   └── geographic.py           # Geographic analysis at WHO regional level (Section 5.8)
│
└── pipeline.py                 # Full pipeline orchestrator + unsupervised scoring weights
```

## Key Pre-Registered Elements in the Code

| Pre-Registration Section | Implementation |
|--------------------------|----------------|
| 5.2 Classifier hyperparameters | `classifier.py` lines 88–104: RF(200 trees, max_depth=10), XGBoost(200 trees, max_depth=6, lr=0.1) |
| 5.4 Threshold selection | `classifier.py`: F1-optimal on PR curve |
| 5.5 PU Learning (e3 method) | `pu_learning.py`: `estimate_label_frequency_cv()` with method="e3" |
| 4.2 Feature exclusion (cited_by_count) | `feature_builder.py` line 47: documented exclusion |
| Appendix A: Unsupervised weights | `pipeline.py`: `_unsupervised_score()` with frozen weights |
| 5.10.2 Geographic sensitivity | `geographic.py`: `geographic_sensitivity_analysis()` |

## Dependencies

```
pyalex>=0.14
requests>=2.31
lxml>=5.0
pandas>=2.0
scikit-learn>=1.4
xgboost>=2.0
numpy>=1.26
```

## Licence

This code is shared for transparency and reproducibility. Full licence terms will be applied at manuscript submission.
