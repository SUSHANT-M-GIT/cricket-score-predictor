"""
train_classification.py
-----------------------
Trains the winner-prediction (classification) models on 2nd-innings data:
  • Logistic Regression  (baseline)
  • Random Forest Classifier (comparison)

Target label : 1 = batting team wins, 0 = bowling team wins.

Evaluates with:
  • Accuracy
  • Confusion Matrix
  • Classification Report (Precision / Recall / F1)

Saves the best model to model/classifier.pkl.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_data, clean_data, encode_teams, get_encoders
from features import engineer_features, get_feature_matrix, CLASSIFICATION_FEATURES

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR        = os.path.dirname(__file__)
DATA_PATH       = os.path.join(_SRC_DIR, "..", "data", "matches.csv")
MODEL_DIR       = os.path.join(_SRC_DIR, "..", "model")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
def _make_winner_label(df: pd.DataFrame) -> pd.Series:
    """
    Binary label: 1 = batting team wins the match, 0 = bowling team wins.
    The 'winner' column contains the winning team name (string).
    """
    return (df["winner"] == df["batting_team_name"]).astype(int)


def _evaluate_classifier(name: str, model, X_test, y_test, scaler=None) -> dict:
    X_input = scaler.transform(X_test) if scaler else X_test
    preds   = model.predict(X_input)
    probs   = model.predict_proba(X_input)[:, 1]
    acc     = accuracy_score(y_test, preds)
    cm      = confusion_matrix(y_test, preds)
    report  = classification_report(y_test, preds, target_names=["Bowling Team Wins", "Batting Team Wins"])

    print(f"\n  ┌── {name}")
    print(f"  │   Accuracy : {acc * 100:.2f}%")
    print(f"  └── Confusion Matrix:\n")
    print("       Predicted: Lose  Win")
    for i, row in enumerate(cm):
        label = "Actual Lose:" if i == 0 else "Actual Win :"
        print(f"       {label}  {row[0]:4d}  {row[1]:4d}")
    print(f"\n{report}")

    return {
        "name": name, "model": model, "scaler": scaler,
        "preds": preds, "probs": probs,
        "acc": acc, "cm": cm,
    }


def _plot_confusion_matrices(results: list, labels: list[str]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    fig.suptitle("Classification: Confusion Matrices", fontsize=13, fontweight="bold")

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res["cm"],
            display_labels=labels,
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{res['name']}\nAccuracy={res['acc']*100:.1f}%")

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Confusion matrix chart → {path}")
    plt.close()


def _plot_feature_importance_clf(model: RandomForestClassifier) -> None:
    imp    = model.feature_importances_
    idx    = np.argsort(imp)[::-1]
    feats  = [CLASSIFICATION_FEATURES[i] for i in idx]
    scores = imp[idx]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=scores, y=feats, hue=feats, palette="rocket", legend=False)
    plt.title("RF Classifier – Feature Importances (Winner Prediction)")
    plt.xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "classifier_feature_importance.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Classifier feature importance chart → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
def train(csv_path: str = DATA_PATH, visualize: bool = True) -> None:
    print("=" * 58)
    print("  CRICKET PREDICTOR – CLASSIFICATION MODEL TRAINING")
    print("=" * 58)

    # ── Load full dataset ──────────────────────────────────────────────────
    df_raw = load_data(csv_path)
    df_raw = clean_data(df_raw)

    # We need the original team name to build the label BEFORE encoding
    df_raw["batting_team_name"] = df_raw["batting_team"]

    # Use the encoders already saved by train_regression.py (fit=False)
    df_enc, encoders = encode_teams(df_raw, fit=False)

    # ── Keep only 2nd-innings rows ─────────────────────────────────────────
    if "innings" not in df_enc.columns:
        raise ValueError(
            "Dataset must have an 'innings' column (1 or 2). "
            "Re-run data/generate_sample_data.py."
        )
    df2 = df_enc[df_enc["innings"] == 2].copy()
    print(f"[i] 2nd-innings rows for classification : {len(df2)}")

    # ── Build label ────────────────────────────────────────────────────────
    df2 = df2.reset_index(drop=True)
    y = _make_winner_label(df2)
    print(f"[i] Label balance – Batting wins: {y.sum()}  |  Bowling wins: {(y == 0).sum()}")

    # ── Feature engineering ────────────────────────────────────────────────
    df2 = engineer_features(df2)
    X, _ = get_feature_matrix(df2, CLASSIFICATION_FEATURES)
    print(f"[i] Features : {list(X.columns)}")

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[i] Train / Test : {len(X_train)} / {len(X_test)}")

    # ── Scale for Logistic Regression ──────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Train models ───────────────────────────────────────────────────────
    print("\n[*] Training Logistic Regression …")
    lr_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_clf.fit(X_train_sc, y_train)

    print("[*] Training Random Forest Classifier …")
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n── Classification Evaluation ─────────────────────────")
    lr_res = _evaluate_classifier(
        "Logistic Regression", lr_clf, X_test, y_test, scaler=scaler
    )
    rf_res = _evaluate_classifier(
        "Random Forest Classifier", rf_clf, X_test, y_test, scaler=None
    )

    # ── Pick best (higher accuracy) ────────────────────────────────────────
    best = rf_res if rf_res["acc"] >= lr_res["acc"] else lr_res
    # Store scaler alongside LR model so predict.py can apply it
    save_bundle = {
        "model": best["model"],
        "scaler": best["scaler"],   # None for RF, StandardScaler for LR
        "model_name": best["name"],
    }
    print(f"\n[✓] Best classifier : {best['name']}  (Acc={best['acc']*100:.2f}%)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(save_bundle, f)
    # Save scaler separately too (for safe reuse)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[✓] Classifier saved → {CLASSIFIER_PATH}")

    if visualize:
        print("\n[*] Generating charts …")
        _plot_confusion_matrices(
            [lr_res, rf_res],
            labels=["Bowling Wins", "Batting Wins"],
        )
        _plot_feature_importance_clf(rf_clf)

    print("\n" + "=" * 58)
    print("  Classification training complete.")
    print("=" * 58)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
