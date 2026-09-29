# Fraud detection: reproducible ML baseline

An original portfolio demonstration using **synthetic data**, not an employer project or production fraud system. Explores imbalanced classification, precision–recall tradeoffs, and validation-only threshold selection.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m fraud_detection.train --output results/baseline.json
```

The script trains class-weighted logistic regression with scaling, using stratified **60% train / 20% validation / 20% untouched test** splits. The threshold is chosen only on validation data to maximize recall subject to minimum precision; the test set is evaluated once. Validation precision does not guarantee test precision.

## Design choices

- **Average precision** summarizes precision–recall behavior for rare positives.
- **Validation-only thresholding** prevents selecting a threshold using test labels.
- **Class-weighted logistic regression** provides an interpretable baseline.
- **Fixed seed and CI tests** make experiments reproducible.

## Baseline results

With seed 42 and 12,000 synthetic samples, validation minimum precision was 0.30. Held-out test: **average precision 0.299**, **precision 0.414**, **recall 0.324**, **17 false positives** and **25 false negatives**. Results are from one synthetic run, not a claim about real-world fraud detection; see `results/baseline.json`.

## Limitations and next steps

Synthetic labels are not real fraud. This baseline has no temporal split, calibration guarantee, drift monitoring, real intervention costs, or production API. Future experiments: unweighted baseline, calibration, cost-sensitive thresholds and uncertainty across seeds. These are planned, not implemented.
