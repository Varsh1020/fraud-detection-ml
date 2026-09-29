"""Train and evaluate an imbalanced-classification baseline on synthetic data."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def generate_data(n_samples=12000, seed=42):
    """Synthetic data only: no customer, transaction, or employer data."""
    return make_classification(
        n_samples=n_samples, n_features=20, n_informative=8,
        n_redundant=4, weights=[0.985, 0.015], flip_y=0.002,
        class_sep=1.8, random_state=seed,
    )


def select_threshold(y_true, scores, min_precision=0.30):
    """Maximize recall subject to minimum precision on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.flatnonzero(precision[:-1] >= min_precision)
    if len(valid) == 0:
        return 1.0
    best = valid[np.argmax(recall[valid])]
    return float(thresholds[best])


def metrics_at_threshold(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "false_positives": int(fp), "false_negatives": int(fn),
        "true_positives": int(tp), "true_negatives": int(tn),
    }


def run(seed=42, n_samples=12000, min_precision=0.30):
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")
    if n_samples < 1000:
        raise ValueError("n_samples must be at least 1000")
    X, y = generate_data(n_samples, seed)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, stratify=y_holdout, random_state=seed
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_scores, min_precision)
    test_scores = model.predict_proba(X_test)[:, 1]
    return {
        "dataset": "synthetic sklearn make_classification",
        "seed": seed, "samples": n_samples,
        "class_prevalence": float(y.mean()),
        "split": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "minimum_validation_precision": min_precision,
        "threshold_selected_on_validation": threshold,
        "validation": metrics_at_threshold(y_val, val_scores, threshold),
        "test": metrics_at_threshold(y_test, test_scores, threshold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--min-precision", type=float, default=0.30)
    parser.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    args = parser.parse_args()
    result = run(args.seed, args.samples, args.min_precision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
