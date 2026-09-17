"""
train_regression.py
-------------------
Trains the score-prediction (regression) models:
  • Linear Regression   (baseline)
  • Random Forest Regressor (comparison)

Evaluates both with MAE and R², saves the best to model/model.pkl.
Also generates feature-importance and predictions-vs-actual charts.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_data, clean_data, encode_teams
from features import engineer_features, get_feature_matrix, REGRESSION_FEATURES

# -- Paths ---------------------------------------------------------------------
_SRC_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(_SRC_DIR, "..", "data", "matches.csv")
MODEL_DIR = os.path.join(_SRC_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# -- Hyper-parameters ----------------------------------------------------------
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
RF_ESTIMATORS  = 200
RF_MAX_DEPTH   = 12


# -----------------------------------------------------------------------------
def _evaluate(name: str, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"\n  +-- {name}")
    print(f"  |   MAE  : {mae:.2f} runs")
    print(f"  +-- R²   : {r2:.4f}")
    return {"name": name, "model": model, "preds": preds, "mae": mae, "r2": r2}


def _plot_importance(model: RandomForestRegressor) -> None:
    imp     = model.feature_importances_
    idx     = np.argsort(imp)[::-1]
    feats   = [REGRESSION_FEATURES[i] for i in idx]
    scores  = imp[idx]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=scores, y=feats, hue=feats, palette="viridis", legend=False)
    plt.title("RF Regressor – Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    print(f"[OK] Feature importance chart -> {path}")
    plt.close()


def _plot_predictions(results: list, y_test: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Regression: Predicted vs Actual Final Score",
                 fontsize=13, fontweight="bold")
    for ax, res in zip(axes, results):
        ax.scatter(y_test, res["preds"], alpha=0.35, s=16, color="#1f77b4")
        lo = min(y_test.min(), res["preds"].min()) - 5
        hi = max(y_test.max(), res["preds"].max()) + 5
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Perfect fit")
        ax.set(xlim=(lo, hi), ylim=(lo, hi),
               xlabel="Actual", ylabel="Predicted",
               title=f"{res['name']}\nMAE={res['mae']:.1f}  R²={res['r2']:.3f}")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "predictions_vs_actual.png")
    plt.savefig(path, dpi=150)
    print(f"[OK] Predictions-vs-actual chart -> {path}")
    plt.close()


# -----------------------------------------------------------------------------
def train(csv_path: str = DATA_PATH, visualize: bool = True) -> None:
    print("=" * 58)
    print("  CRICKET PREDICTOR – REGRESSION MODEL TRAINING")
    print("=" * 58)

    df = load_data(csv_path)
    df = clean_data(df)
    df, _ = encode_teams(df, fit=True)   # fits + saves encoders.pkl
    df = engineer_features(df)

    X, y = get_feature_matrix(df, REGRESSION_FEATURES, "final_score")
    print(f"\n[i] Features : {list(X.columns)}")
    print(f"[i] Samples  : {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[i] Train / Test split : {len(X_train)} / {len(X_test)}")

    print("\n[*] Training Linear Regression ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print("[*] Training Random Forest Regressor ...")
    rf = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    print("\n-- Regression Evaluation -----------------------------")
    lr_res = _evaluate("Linear Regression",       lr, X_test, y_test)
    rf_res = _evaluate("Random Forest Regressor", rf, X_test, y_test)

    best = rf_res if rf_res["mae"] <= lr_res["mae"] else lr_res
    print(f"\n[OK] Best regression model : {best['name']}  (MAE={best['mae']:.2f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best["model"], f)
    print(f"[OK] Saved -> {MODEL_PATH}")

    if visualize:
        print("\n[*] Generating charts ...")
        _plot_importance(rf)
        _plot_predictions([lr_res, rf_res], y_test)

    print("\n" + "=" * 58)
    print("  Regression training complete.")
    print("=" * 58)
    return best["mae"]   # returned so orchestrator can store margin


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train()
