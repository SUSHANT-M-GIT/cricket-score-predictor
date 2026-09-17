"""
features.py
-----------
Feature engineering for both innings.

1st innings  ->  score projection features only
2nd innings  ->  adds chase-specific features:
                runs_required, balls_remaining,
                required_run_rate, pressure_index
"""

import pandas as pd
import numpy as np

TOTAL_OVERS = 20.0
AVERAGE_T20_RR = 7.5

# -- Feature sets --------------------------------------------------------------

# Used by the regression model (score prediction) – works for both innings
REGRESSION_FEATURES = [
    "batting_team",
    "bowling_team",
    "current_score",
    "overs_completed",
    "wickets_fallen",
    "current_run_rate",
    "remaining_overs",        # engineered
    "wickets_in_hand",        # engineered
    "projected_score",        # engineered: CRR x 20
    "scoring_rate_factor",    # engineered: CRR / avg T20 RR
]

# Used by the classification model (winner prediction) – 2nd innings only
CLASSIFICATION_FEATURES = [
    "batting_team",
    "bowling_team",
    "current_score",
    "overs_completed",
    "wickets_fallen",
    "current_run_rate",
    "remaining_overs",
    "wickets_in_hand",
    "projected_score",
    "scoring_rate_factor",
    "runs_required",          # NEW: runs still needed to win
    "balls_remaining",        # NEW: balls left in innings
    "required_run_rate",      # NEW: RRR = runs_required / remaining_overs
    "pressure_index",         # NEW: RRR / CRR  (>1 = under pressure)
]

# Backward-compat alias (train.py / train_regression.py reference this)
FEATURE_COLUMNS = REGRESSION_FEATURES


# -- Core engineering ----------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived columns in-place on a copy.
    Safe to call on both 1st- and 2nd-innings data.
    """
    df = df.copy()

    # --- shared features (both innings) ---
    df["remaining_overs"] = (TOTAL_OVERS - df["overs_completed"]).clip(lower=0)
    df["wickets_in_hand"] = (10 - df["wickets_fallen"]).clip(lower=0, upper=10)
    df["projected_score"] = (df["current_run_rate"] * TOTAL_OVERS).round(1)
    df["scoring_rate_factor"] = (df["current_run_rate"] / AVERAGE_T20_RR).round(4)

    # --- 2nd-innings chase features ---
    # These are zero / meaningless for 1st innings but won't break training
    # because the regression model doesn't use them.
    if "target" in df.columns:
        df["runs_required"] = (df["target"] - df["current_score"]).clip(lower=0)
        df["balls_remaining"] = (df["remaining_overs"] * 6).astype(int)
        df["required_run_rate"] = (
            df["runs_required"] / df["remaining_overs"].replace(0, np.nan)
        ).fillna(99.0).round(2)
        df["pressure_index"] = (
            df["required_run_rate"] / df["current_run_rate"].replace(0, np.nan)
        ).fillna(99.0).round(4)
    else:
        # Ensure columns exist with neutral values for prediction rows
        for col in ("runs_required", "balls_remaining", "required_run_rate", "pressure_index"):
            if col not in df.columns:
                df[col] = 0.0

    return df


# -- Feature matrix helpers ----------------------------------------------------

def get_feature_matrix(
    df: pd.DataFrame,
    feature_set: list[str] = REGRESSION_FEATURES,
    target_col: str = "final_score",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (X, y). y is None when target_col is absent (inference mode)."""
    missing = [c for c in feature_set if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing features {missing}. "
            "Call engineer_features() before get_feature_matrix()."
        )
    X = df[feature_set]
    y = df[target_col] if target_col in df.columns else None
    return X, y


# -- Single-row builders for inference ----------------------------------------

def build_regression_row(
    current_score: int,
    overs_completed: float,
    wickets_fallen: int,
    batting_team_enc: int,
    bowling_team_enc: int,
    current_run_rate: float,
) -> pd.DataFrame:
    """Feature row for the score-regression model (works for both innings)."""
    raw = pd.DataFrame([{
        "batting_team": batting_team_enc,
        "bowling_team": bowling_team_enc,
        "current_score": current_score,
        "overs_completed": overs_completed,
        "wickets_fallen": wickets_fallen,
        "current_run_rate": current_run_rate,
    }])
    enriched = engineer_features(raw)
    X, _ = get_feature_matrix(enriched, REGRESSION_FEATURES)
    return X


def build_classification_row(
    current_score: int,
    overs_completed: float,
    wickets_fallen: int,
    batting_team_enc: int,
    bowling_team_enc: int,
    current_run_rate: float,
    target: int,
) -> pd.DataFrame:
    """Feature row for the winner-classification model (2nd innings only)."""
    raw = pd.DataFrame([{
        "batting_team": batting_team_enc,
        "bowling_team": bowling_team_enc,
        "current_score": current_score,
        "overs_completed": overs_completed,
        "wickets_fallen": wickets_fallen,
        "current_run_rate": current_run_rate,
        "target": target,
    }])
    enriched = engineer_features(raw)
    X, _ = get_feature_matrix(enriched, CLASSIFICATION_FEATURES)
    return X


# -- Keep old name as alias so existing callers don't break -------------------
build_prediction_row = build_regression_row
