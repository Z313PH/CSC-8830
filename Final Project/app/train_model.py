"""
train_model.py — Train a scikit-learn gesture classifier on collected landmarks.

Usage:
    python train_model.py                   # auto-finds all CSVs in data/
    python train_model.py --data data/landmarks_20240101_120000.csv

Outputs:
    models/model.joblib       — trained Pipeline (scaler + classifier)
    models/confusion_matrix.png
    models/training_report.txt
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")

TEST_SIZE   = 0.20
RANDOM_SEED = 42


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    csvs = sorted(data_dir.glob("landmarks_*.csv"))
    if not csvs:
        print(f"[ERROR] No landmark CSVs found in {data_dir}/")
        sys.exit(1)
    print(f"[DATA] Loading {len(csvs)} CSV file(s):")
    frames = []
    for f in csvs:
        df = pd.read_csv(f)
        print(f"       {f.name}  ({len(df)} rows)")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def prepare_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y


# ──────────────────────────────────────────────
# 2. CLASSIFIER DEFINITIONS + HYPERPARAMETER GRIDS
# ──────────────────────────────────────────────

def get_classifier_options():
    """
    Returns a dict of {name: (estimator, param_grid)} for GridSearchCV.
    Adjust search spaces as needed.
    """
    return {
        "svm": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED),
            {
                "clf__C":     [0.1, 1.0, 10.0, 100.0],
                "clf__gamma": ["scale", "auto"],
                "clf__kernel": ["rbf"],
            },
        ),
        "logistic": (
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                n_jobs=-1,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth":    [None, 10, 20],
            },
        ),
    }


# ──────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────

def build_and_search(clf_name: str, X_train, y_train):
    options = get_classifier_options()
    if clf_name not in options:
        raise ValueError(f"Unknown classifier '{clf_name}'. Choose from: {list(options.keys())}")

    estimator, param_grid = options[clf_name]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    estimator),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    print(f"\n[TRAIN] Running GridSearchCV for '{clf_name}'...")
    grid.fit(X_train, y_train)

    print(f"[TRAIN] Best CV accuracy: {grid.best_score_:.4f}")
    print(f"[TRAIN] Best params:      {grid.best_params_}")
    return grid.best_estimator_


# ──────────────────────────────────────────────
# 4. EVALUATION
# ──────────────────────────────────────────────

def evaluate(model, X_test, y_test, class_names, output_dir: Path):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=class_names)
    cm     = confusion_matrix(y_test, y_pred, labels=class_names)

    print(f"\n{'='*52}")
    print(f"  TEST ACCURACY: {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*52}")
    print(report)

    # Save text report
    report_path = output_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"[SAVED] Report → {report_path}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix  (test acc = {acc:.3f})", fontsize=13)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] Confusion matrix → {cm_path}")

    return acc


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train gesture classifier.")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to a specific CSV file. Default: merge all CSVs in data/",
    )
    parser.add_argument(
        "--clf", type=str, default="svm",
        choices=["svm", "logistic", "random_forest"],
        help="Classifier to train (default: svm)",
    )
    args = parser.parse_args()

    # Load data
    if args.data:
        df = pd.read_csv(args.data)
        print(f"[DATA] Loaded {len(df)} rows from {args.data}")
    else:
        df = load_all_csvs(DATA_DIR)

    print(f"\n[DATA] Total samples: {len(df)}")
    print(f"[DATA] Class distribution:\n{df['label'].value_counts().to_string()}\n")

    # Check minimum samples
    min_count = df["label"].value_counts().min()
    if min_count < 20:
        print(f"[WARNING] Some classes have very few samples ({min_count}). "
              "Collect more data for reliable training.")

    X, y = prepare_xy(df)
    class_names = sorted(df["label"].unique().tolist())
    print(f"[DATA] Classes ({len(class_names)}): {class_names}")
    print(f"[DATA] Feature vector size: {X.shape[1]}\n")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    best_model = build_and_search(args.clf, X_train, y_train)

    # Evaluate
    evaluate(best_model, X_test, y_test, class_names, MODELS_DIR)

    # Save model
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\n[SAVED] Model → {model_path}")
    print(f"\n[DONE] Run `python run_live_demo.py` to test live.")


if __name__ == "__main__":
    main()
