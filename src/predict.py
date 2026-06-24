"""
predict.py
----------
Unified prediction module.

predict_score()   → regression  (both innings)
predict_winner()  → classification (2nd innings only)
predict_full()    → both combined  (2nd innings)
"""

import os
import pickle
import numpy as np
import pandas as pd

from preprocess import get_encoders
from features import build_regression_row, build_classification_row

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR        = os.path.dirname(__file__)
MODEL_PATH      = os.path.join(_SRC_DIR, "..", "model", "model.pkl")
CLASSIFIER_PATH = os.path.join(_SRC_DIR, "..", "model", "classifier.pkl")

DEFAULT_MARGIN  = 10   # ± runs shown in confidence range


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_regressor():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No regression model at '{MODEL_PATH}'.\n"
            "Run  python main.py --train  first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _load_classifier() -> dict:
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"No classifier at '{CLASSIFIER_PATH}'.\n"
            "Run  python main.py --train  first."
        )
    with open(CLASSIFIER_PATH, "rb") as f:
        return pickle.load(f)   # {"model": ..., "scaler": ..., "model_name": ...}


def _encode_teams(batting_team: str, bowling_team: str) -> tuple[int, int]:
    """Encode team names using the saved LabelEncoders."""
    encoders = get_encoders()

    def _enc(col: str, val: str) -> int:
        le = encoders[col]
        known = set(le.classes_)
        v = val if val in known else le.classes_[0]
        return int(le.transform([v])[0])

    return _enc("batting_team", batting_team), _enc("bowling_team", bowling_team)


# ── Public API ────────────────────────────────────────────────────────────────

def predict_score(
    current_score: int,
    overs_completed: float,
    wickets_fallen: int,
    batting_team: str,
    bowling_team: str,
    current_run_rate: float,
    margin: int = DEFAULT_MARGIN,
) -> dict:
    """
    Predict the final score (works for both 1st and 2nd innings).

    Returns
    -------
    {
        predicted_score, lower_bound, upper_bound,
        remaining_overs, wickets_in_hand, projected_score_crr
    }
    """
    model = _load_regressor()
    bat_enc, bowl_enc = _encode_teams(batting_team, bowling_team)

    X = build_regression_row(
        current_score=current_score,
        overs_completed=overs_completed,
        wickets_fallen=wickets_fallen,
        batting_team_enc=bat_enc,
        bowling_team_enc=bowl_enc,
        current_run_rate=current_run_rate,
    )

    predicted       = int(round(model.predict(X)[0]))
    remaining_overs = round(20.0 - overs_completed, 1)
    wickets_in_hand = 10 - wickets_fallen

    return {
        "predicted_score":    predicted,
        "lower_bound":        max(current_score, predicted - margin),
        "upper_bound":        predicted + margin,
        "remaining_overs":    remaining_overs,
        "wickets_in_hand":    wickets_in_hand,
        "projected_score_crr": int(current_run_rate * 20),
    }


def predict_winner(
    current_score: int,
    overs_completed: float,
    wickets_fallen: int,
    batting_team: str,
    bowling_team: str,
    current_run_rate: float,
    target: int,
) -> dict:
    """
    Predict the likely match winner (2nd innings only).

    Returns
    -------
    {
        winner, win_probability, lose_probability,
        batting_team, bowling_team,
        runs_required, balls_remaining, required_run_rate, pressure_index
    }
    """
    bundle  = _load_classifier()
    clf     = bundle["model"]
    scaler  = bundle["scaler"]
    bat_enc, bowl_enc = _encode_teams(batting_team, bowling_team)

    X = build_classification_row(
        current_score=current_score,
        overs_completed=overs_completed,
        wickets_fallen=wickets_fallen,
        batting_team_enc=bat_enc,
        bowling_team_enc=bowl_enc,
        current_run_rate=current_run_rate,
        target=target,
    )

    X_input  = scaler.transform(X) if scaler is not None else X
    proba    = clf.predict_proba(X_input)[0]   # [P(bowl wins), P(bat wins)]
    bat_wins = bool(proba[1] >= 0.5)
    winner   = batting_team if bat_wins else bowling_team
    win_prob = proba[1] if bat_wins else proba[0]

    remaining_overs  = round(20.0 - overs_completed, 1)
    runs_required    = max(0, target - current_score)
    balls_remaining  = int(remaining_overs * 6)
    rrr = round(runs_required / remaining_overs, 2) if remaining_overs > 0 else 99.0
    pressure = round(rrr / current_run_rate, 4) if current_run_rate > 0 else 99.0

    return {
        "winner":            winner,
        "win_probability":   round(float(win_prob) * 100, 1),
        "batting_team_prob": round(float(proba[1]) * 100, 1),
        "bowling_team_prob": round(float(proba[0]) * 100, 1),
        "batting_team":      batting_team,
        "bowling_team":      bowling_team,
        "runs_required":     runs_required,
        "balls_remaining":   balls_remaining,
        "required_run_rate": rrr,
        "pressure_index":    pressure,
    }


def predict_full(
    current_score: int,
    overs_completed: float,
    wickets_fallen: int,
    batting_team: str,
    bowling_team: str,
    current_run_rate: float,
    target: int,
    margin: int = DEFAULT_MARGIN,
) -> dict:
    """
    Combined prediction for 2nd innings: score + winner.
    Returns merged dict from predict_score and predict_winner.
    """
    score_result  = predict_score(
        current_score, overs_completed, wickets_fallen,
        batting_team, bowling_team, current_run_rate, margin,
    )
    winner_result = predict_winner(
        current_score, overs_completed, wickets_fallen,
        batting_team, bowling_team, current_run_rate, target,
    )
    return {**score_result, **winner_result}
