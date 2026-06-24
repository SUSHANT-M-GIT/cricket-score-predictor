"""
preprocess.py
-------------
Handles loading the raw CSV dataset, cleaning nulls,
and encoding categorical columns (team names).
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle


# Columns expected in the raw CSV
REQUIRED_COLUMNS = [
    "batting_team", "bowling_team", "overs_completed",
    "wickets_fallen", "current_score", "current_run_rate",
]
TARGET_COLUMN = "final_score"

# Path where encoders are persisted so prediction uses the same mapping
ENCODER_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "encoders.pkl"
)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and perform basic validation."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Run  python data/generate_sample_data.py  to create one."
        )

    df = pd.read_csv(csv_path)
    print(f"[✓] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    missing = [c for c in REQUIRED_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls and remove obviously invalid rows."""
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS + [TARGET_COLUMN])

    # Sanity guards
    df = df[df["overs_completed"] > 0]
    df = df[df["wickets_fallen"].between(0, 10)]
    df = df[df["current_score"] >= 0]
    df = df[df["final_score"] >= df["current_score"]]

    after = len(df)
    if before != after:
        print(f"[i] Dropped {before - after} invalid/null rows. Remaining: {after}")
    return df.reset_index(drop=True)


def encode_teams(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode 'batting_team' and 'bowling_team'.

    Parameters
    ----------
    df       : DataFrame to encode (modified in place on a copy)
    fit      : If True, fit new encoders; if False, use provided encoders.
    encoders : Pre-fitted encoders to reuse during prediction.

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()

    if fit:
        encoders = {}
        for col in ("batting_team", "bowling_team"):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        _save_encoders(encoders)
        print("[✓] Team encoders fitted and saved.")
    else:
        if encoders is None:
            encoders = _load_encoders()
        for col in ("batting_team", "bowling_team"):
            le: LabelEncoder = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def _save_encoders(encoders: dict) -> None:
    os.makedirs(os.path.dirname(ENCODER_SAVE_PATH), exist_ok=True)
    with open(ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(encoders, f)


def _load_encoders() -> dict:
    if not os.path.exists(ENCODER_SAVE_PATH):
        raise FileNotFoundError(
            f"Encoders not found at '{ENCODER_SAVE_PATH}'. Train the model first."
        )
    with open(ENCODER_SAVE_PATH, "rb") as f:
        return pickle.load(f)


def get_encoders() -> dict:
    """Public helper to retrieve saved encoders (used by predict.py)."""
    return _load_encoders()
