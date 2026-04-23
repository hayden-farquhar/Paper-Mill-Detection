# Changelog

## v3.0.0 (2026-04-24) — Post-revision release

Code update following manuscript revision after peer review at Accountability in Research. No changes to pre-registered classifier, weights, or feature definitions.

### Added
- `leave_hindawi_out_sensitivity.py`: Standalone script for the post-hoc leave-Hindawi-out sensitivity analysis. Removes Hindawi-journal papers and re-runs the classifier with identical architecture and hyperparameters. Results: AP 0.437 (vs 0.858 primary), PU-corrected prevalence 15.5% (vs 22.1%).

### No changes to
- Classifier architecture or hyperparameters (pre-registered, frozen)
- Unsupervised scoring weights (pre-registered, frozen)
- Feature definitions or interaction features
- PU learning implementation
- Core pipeline modules in `src/`

## v2.0.0 (2026-04-09) — Post-analysis release

Final code as used in the analyses reported in the manuscript. Changes from v1.0.0 (pre-registered):

### Bug fixes
- `detect/author_network.py`: Fixed `frozenset` unpacking error in `find_author_pools()` for single-element sets
- `detect/bibliometric_flags.py`: Fixed `AttributeError` when `all_countries` field is NaN (cast to string before split)

### Dictionary refinements
- `detect/tortured_phrases.py`: Removed "review study" and "directed learning" as false positives identified during pilot testing, replaced with medical AI-specific entries (94 phrases; documented in manuscript Methods)

### No changes to
- Classifier architecture or hyperparameters (pre-registered, frozen)
- Unsupervised scoring weights (pre-registered, frozen)
- Feature definitions or interaction features
- PU learning implementation
- Analysis modules

## v1.0.0 (2026-04-09) — Pre-registered release

Initial code deposit, registered on OSF prior to classifier training or inferential analysis.
DOI: 10.5281/zenodo.19481250
