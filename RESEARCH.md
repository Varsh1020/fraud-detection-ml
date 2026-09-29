# Research protocol: evaluation under class imbalance

**Status:** Independent portfolio research plan, started September 2026. This repository is **not** presented as a university-approved thesis, an MS course submission, or six months of completed research.

## Research question

How stable are precision-constrained decision thresholds for imbalanced binary classification across random seeds, class prevalence shifts, and changes in the relative cost of false positives and false negatives?

## Baseline

The implemented baseline uses synthetic `make_classification` data, stratified 60/20/20 train/validation/test splits, standardized features, and class-weighted logistic regression. A decision threshold is selected on validation data and evaluated on an untouched test split. See [README](README.md) and [baseline results](results/baseline.json).

## Planned experiments (not yet performed)

1. **Repeated seeds:** Repeat the pipeline across at least 20 seeds; report mean, standard deviation, and confidence intervals for average precision, precision, recall, and false-positive counts.
2. **Baseline comparison:** Compare class-weighted and unweighted logistic regression using identical splits and feature processing.
3. **Prevalence sensitivity:** Evaluate on synthetic datasets with different positive-class prevalences while keeping the data-generation settings documented.
4. **Calibration:** Compare raw scores and calibrated probabilities; report Brier score and reliability diagrams on held-out data.
5. **Decision costs:** Evaluate a range of explicit false-positive/false-negative cost ratios rather than claiming one universal optimal threshold.

## Evaluation safeguards

- Select model hyperparameters and thresholds without using test labels.
- Report both PR-AUC (average precision) and operating-point metrics.
- Report uncertainty and class counts; avoid cherry-picking the best seed.
- Record package versions and configuration for reproducibility.
- Clearly separate implemented results from hypotheses and future work.

## Threats to validity

Synthetic labels do not represent real-world fraud. Random stratified splits may overestimate performance under temporal drift. Precision depends on class prevalence. Validation threshold constraints may fail to transfer to test or deployment distributions.

## Research log

| Date | Milestone | Evidence |
| --- | --- | --- |
| 2026-09-29 | Published initial synthetic-data baseline, tests, and CI | Repository commit history |
| 2026-09-29 | Added prospective research protocol | This document |

New milestones should be recorded when the corresponding code and results are actually published.
