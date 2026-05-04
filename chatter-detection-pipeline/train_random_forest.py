"""
Balanced Random Forest exploratory feature validation.

Purpose
-------
This script tests whether the extracted features can separate stable operation
from synthetic injected chatter-like intervals.

Important:
- It does NOT use detector_state labels.
- Real runs are labelled negative.
- Synthetic runs are labelled positive only inside the injected chatter window.
- The benchmark is balanced by sampling a controlled number of negative frames.
- Therefore this is an exploratory feature-validation test, not an operational
  ML detector validation.

Run after:
    python main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
RESULTS_BY_RUN = BASE_DIR / "results_by_run"
SYNTH_LABEL_DIR = BASE_DIR / "outputs" / "synthetic" / "labels"
OUT_DIR = BASE_DIR / "models" / "rf_balanced_independent_labels"

RANDOM_SEED = 42

REAL_RUNS = {"roll1", "roll2", "roll3"}
SYNTH_RUNS = {"synthetic_roll1", "synthetic_roll2", "synthetic_roll3"}

# Negative-to-positive ratio for the balanced RF benchmark.
# 5 means: all positives + 5x as many negatives.
NEGATIVE_TO_POSITIVE_RATIO = 5

# Used only if no injection-label CSV is found.
# This matches the synthetic injection script:
# third-band injection starts at max(20, 0.25 * duration)
# onset = 3 s, steady = 6 s.
INJECTION_START_FRACTION = 0.25
INJECTION_ONSET_SEC = 3.0
INJECTION_STEADY_SEC = 6.0

# Padding accounts for overlap of 2 s analysis windows.
LABEL_PADDING_SEC = 1.5


FEATURE_COLS = [
    "vib_rms_os",
    "vib_rms_ds",
    "vib_rms_mean",
    "rms_ratio_os_ds",
    "corr_os_ds",
    "coh_third",
    "coh_fifth",
    "spec_flatness",
    "kurtosis",
    "skewness",
    "e_wide",
    "e_third",
    "e_fifth",
    "alpha_third",
    "alpha_fifth",
    "growth_third",
    "growth_fifth",
    "dom_third_hz",
    "dom_fifth_hz",
    "third_valid",
    "fifth_valid",
    "third_energy_ratio",
    "fifth_energy_ratio",
    "third_prominence_ratio",
    "fifth_prominence_ratio",
    "third_snr_db",
    "fifth_snr_db",
]


def discover_feature_files() -> list[tuple[str, Path]]:
    if not RESULTS_BY_RUN.is_dir():
        raise FileNotFoundError(
            f"Missing folder: {RESULTS_BY_RUN}\n"
            "Run main.py first so results_by_run/*/features.csv exists."
        )

    found: list[tuple[str, Path]] = []

    for run_dir in sorted(RESULTS_BY_RUN.iterdir()):
        if not run_dir.is_dir():
            continue

        features_path = run_dir / "features.csv"
        if features_path.exists():
            found.append((run_dir.name, features_path))

    if not found:
        raise FileNotFoundError("No features.csv files found inside results_by_run.")

    return found


def choose_feature_columns(df: pd.DataFrame) -> list[str]:
    usable: list[str] = []
    dropped_missing: list[str] = []
    dropped_nan: list[str] = []

    for col in FEATURE_COLS:
        if col not in df.columns:
            dropped_missing.append(col)
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            usable.append(col)
        else:
            dropped_nan.append(col)

    if dropped_missing:
        print(f"Dropping missing features: {', '.join(dropped_missing)}")

    if dropped_nan:
        print(f"Dropping all-NaN features: {', '.join(dropped_nan)}")

    if not usable:
        raise RuntimeError("No usable feature columns found.")

    return usable


def infer_duration_sec(df: pd.DataFrame) -> float:
    if "t_sec" not in df.columns:
        raise ValueError("features.csv must contain t_sec for independent labels.")

    t = pd.to_numeric(df["t_sec"], errors="coerce").dropna()
    if len(t) == 0:
        raise ValueError("t_sec column has no usable numeric values.")

    return float(t.max())


def label_csv_candidates(run_name: str) -> list[Path]:
    """
    Possible locations for independent synthetic label CSVs.

    Expected names from previous injection script:
        validation_labels_synthetic_roll1.csv
        validation_labels_synthetic_roll2.csv
        validation_labels_synthetic_roll3.csv
    """

    suffix = run_name.replace("synthetic_", "")

    return [
        BASE_DIR / f"validation_labels_synthetic_{suffix}.csv",
        SYNTH_LABEL_DIR / f"validation_labels_synthetic_{suffix}.csv",
        BASE_DIR / "outputs" / "synthetic" / "labels" / f"validation_labels_synthetic_{suffix}.csv",
    ]


def labels_from_csv(run_name: str, df: pd.DataFrame) -> tuple[pd.Series, str] | None:
    """
    Try to read injection labels from CSV.

    Required columns:
        start_sec, end_sec, label

    Positive labels:
        onset, chatter, third, fifth
    """

    for path in label_csv_candidates(run_name):
        if not path.exists():
            continue

        labels = pd.read_csv(path)

        required = {"start_sec", "end_sec", "label"}
        if not required.issubset(set(labels.columns)):
            continue

        t = pd.to_numeric(df["t_sec"], errors="coerce")
        y = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

        labels = labels.copy()
        labels["label"] = labels["label"].astype(str).str.lower().str.strip()

        positive_labels = {"onset", "chatter", "third", "fifth"}

        for _, row in labels.iterrows():
            if row["label"] not in positive_labels:
                continue

            start = float(row["start_sec"]) - LABEL_PADDING_SEC
            end = float(row["end_sec"]) + LABEL_PADDING_SEC

            mask = (t >= start) & (t <= end)
            y.loc[mask.fillna(False)] = 1

        return y, f"csv_labels:{path.name}"

    return None


def inferred_synthetic_labels(run_name: str, df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Fallback if no label CSV exists.

    This infers the expected synthetic third-band injection window from the
    injection script convention.
    """

    duration = infer_duration_sec(df)

    start_sec = max(20.0, INJECTION_START_FRACTION * duration)
    end_sec = start_sec + INJECTION_ONSET_SEC + INJECTION_STEADY_SEC

    label_start = max(0.0, start_sec - LABEL_PADDING_SEC)
    label_end = end_sec + LABEL_PADDING_SEC

    t = pd.to_numeric(df["t_sec"], errors="coerce")
    y = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    mask = (t >= label_start) & (t <= label_end)
    y.loc[mask.fillna(False)] = 1

    source = f"inferred_synthetic_window:{label_start:.2f}-{label_end:.2f}s"
    return y, source


def independent_labels_for_run(run_name: str, df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Real runs:
        all frames negative.

    Synthetic runs:
        positive inside independent injection window.
    """

    if "t_sec" not in df.columns:
        raise ValueError(f"{run_name}: features.csv does not contain t_sec.")

    if run_name in REAL_RUNS:
        y = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        return y, "real_stable_all_negative"

    if run_name in SYNTH_RUNS:
        from_csv = labels_from_csv(run_name, df)
        if from_csv is not None:
            return from_csv

        return inferred_synthetic_labels(run_name, df)

    raise ValueError(f"Unknown run name: {run_name}")


def load_all_data() -> tuple[pd.DataFrame, list[dict]]:
    feature_files = discover_feature_files()

    print("\nFound feature files:")
    for run_name, path in feature_files:
        print(f"  {run_name:<20} {path}")

    frames: list[pd.DataFrame] = []
    label_summary: list[dict] = []

    for run_name, path in feature_files:
        if run_name not in REAL_RUNS and run_name not in SYNTH_RUNS:
            print(f"Skipping unknown run folder: {run_name}")
            continue

        df = pd.read_csv(path)
        df = df.copy()
        df["run_name"] = run_name

        y, label_source = independent_labels_for_run(run_name, df)
        df["label"] = y.astype(int)

        n_total = int(len(df))
        n_pos = int(df["label"].sum())
        n_neg = n_total - n_pos

        print("\n" + "=" * 60)
        print(f"Run: {run_name}")
        print(f"Label source: {label_source}")
        print(f"Positive frames: {n_pos} / {n_total} ({100*n_pos/max(n_total, 1):.2f}%)")
        print(f"Negative frames: {n_neg} / {n_total} ({100*n_neg/max(n_total, 1):.2f}%)")

        label_summary.append(
            {
                "run_name": run_name,
                "label_source": label_source,
                "n_total": n_total,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "positive_fraction": float(n_pos / max(n_total, 1)),
            }
        )

        frames.append(df)

    if not frames:
        raise RuntimeError("No valid runs loaded.")

    data = pd.concat(frames, ignore_index=True)
    return data, label_summary


def make_balanced_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all positive frames and sample a controlled number of negative frames.

    This prevents the RF from learning the trivial solution:
        "almost everything is negative".
    """

    positives = data[data["label"] == 1].copy()
    negatives = data[data["label"] == 0].copy()

    n_pos = len(positives)
    if n_pos == 0:
        raise RuntimeError("No positive frames found. Cannot train RF.")

    n_neg_target = min(len(negatives), NEGATIVE_TO_POSITIVE_RATIO * n_pos)

    # Sample negatives with representation from all runs.
    sampled_negatives = (
        negatives
        .groupby("run_name", group_keys=False)
        .apply(
            lambda g: g.sample(
                n=min(len(g), max(1, int(n_neg_target * len(g) / len(negatives)))),
                random_state=RANDOM_SEED,
            )
        )
    )

    # Adjust if rounding sampled slightly too few or too many.
    if len(sampled_negatives) < n_neg_target:
        remaining = negatives.drop(index=sampled_negatives.index, errors="ignore")
        extra_needed = min(n_neg_target - len(sampled_negatives), len(remaining))
        if extra_needed > 0:
            extra = remaining.sample(n=extra_needed, random_state=RANDOM_SEED)
            sampled_negatives = pd.concat([sampled_negatives, extra], ignore_index=False)

    if len(sampled_negatives) > n_neg_target:
        sampled_negatives = sampled_negatives.sample(n=n_neg_target, random_state=RANDOM_SEED)

    balanced = pd.concat([positives, sampled_negatives], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    return balanced


def choose_threshold_on_validation(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Choose a classification threshold using validation data only.

    Returns:
        best_threshold, best_validation_f1
    """

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> float:
    auc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve — Balanced RF Benchmark")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    return auc


def save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> float:
    pr_auc = float(average_precision_score(y_true, y_prob))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Random Forest Precision-Recall Curve — Balanced RF Benchmark")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=200)
    plt.close()

    return pr_auc


def main() -> None:
    print("Random Forest — Balanced Independent Injection-Window Labels")
    print(f"Project root: {BASE_DIR}")

    data, label_summary = load_all_data()
    feature_cols = choose_feature_columns(data)

    print("\n" + "=" * 60)
    print("Full independent-label dataset")
    print("=" * 60)
    print(f"Total frames     : {len(data):,}")
    print(f"Positive frames  : {int(data['label'].sum()):,}")
    print(f"Negative frames  : {int(len(data) - data['label'].sum()):,}")
    print(f"Positive fraction: {100 * data['label'].mean():.2f}%")

    balanced = make_balanced_dataset(data)

    print("\n" + "=" * 60)
    print("Balanced RF benchmark dataset")
    print("=" * 60)
    print(f"Total frames     : {len(balanced):,}")
    print(f"Positive frames  : {int(balanced['label'].sum()):,}")
    print(f"Negative frames  : {int(len(balanced) - balanced['label'].sum()):,}")
    print(f"Positive fraction: {100 * balanced['label'].mean():.2f}%")
    print(f"Features used    : {len(feature_cols)}")

    X = balanced[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = balanced["label"].astype(int)

    # Split into train / validation / test.
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X,
        y,
        balanced[["run_name", "t_sec", "label"]].copy(),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp,
        y_temp,
        meta_temp,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print("\nSplit:")
    print(f"  Train      : {len(X_train):,} frames | positives: {int(y_train.sum())}")
    print(f"  Validation : {len(X_val):,} frames | positives: {int(y_val.sum())}")
    print(f"  Test       : {len(X_test):,} frames | positives: {int(y_test.sum())}")

    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train)
    Xva = imputer.transform(X_val)
    Xte = imputer.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    rf.fit(Xtr, y_train)

    val_prob = rf.predict_proba(Xva)[:, 1]
    test_prob = rf.predict_proba(Xte)[:, 1]

    threshold, val_f1 = choose_threshold_on_validation(y_val.to_numpy(), val_prob)

    test_pred = (test_prob >= threshold).astype(int)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roc_auc = save_roc_curve(y_test.to_numpy(), test_prob, OUT_DIR)
    pr_auc = save_pr_curve(y_test.to_numpy(), test_prob, OUT_DIR)

    metrics = {
        "model": "RandomForestClassifier",
        "benchmark_type": "balanced_exploratory_feature_validation",
        "label_source": "independent_injection_windows_plus_real_stable_negatives",
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "threshold_selected_on_validation_data": threshold,
        "validation_f1_at_selected_threshold": val_f1,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "n_full_frames": int(len(data)),
        "n_full_positive": int(data["label"].sum()),
        "n_full_negative": int(len(data) - data["label"].sum()),
        "n_balanced_frames": int(len(balanced)),
        "n_balanced_positive": int(balanced["label"].sum()),
        "n_balanced_negative": int(len(balanced) - balanced["label"].sum()),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_positive_train": int(y_train.sum()),
        "n_positive_validation": int(y_val.sum()),
        "n_positive_test": int(y_test.sum()),
        "features_used": feature_cols,
        "label_summary": label_summary,
    }

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    predictions = meta_test.copy()
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_prob"] = test_prob
    predictions["y_pred"] = test_pred

    with open(OUT_DIR / "rf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    importances.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    predictions.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    joblib.dump(rf, OUT_DIR / "random_forest_balanced_independent_labels.pkl")
    joblib.dump(imputer, OUT_DIR / "rf_imputer.pkl")

    print("\n" + "=" * 60)
    print("RF RESULTS — BALANCED INDEPENDENT LABELS")
    print("=" * 60)
    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1       : {metrics['f1']:.3f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(f"PR-AUC   : {metrics['pr_auc']:.3f}")
    print(f"Threshold selected on validation set: {threshold:.2f}")
    print(f"Validation F1 at selected threshold : {val_f1:.3f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    print("\nClassification report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    print("\nTop-10 feature importances:")
    for i, row in enumerate(importances.head(10).itertuples(), start=1):
        print(f"  {i:>2}. {row.feature:<30} {row.importance:.4f}")

    print(f"\nSaved RF outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()