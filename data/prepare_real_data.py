"""
prepare_real_data.py
--------------------
Converts the raw Kaggle IPL ball-by-ball dataset into the training CSV
expected by src/preprocess.py and src/train.py.

Raw inputs  (data/raw/)
  deliveries.csv  –  260 K ball-by-ball rows (Kaggle: patrickb1912)
  matches.csv     –  1 095 match-level rows  (same dataset)

Output
  data/matches.csv  –  replaces the synthetic dataset

Schema produced (matches the REQUIRED_COLUMNS in preprocess.py):
  innings, batting_team, bowling_team, overs_completed, wickets_fallen,
  current_score, current_run_rate, remaining_overs, wickets_in_hand,
  target, runs_required, balls_remaining, required_run_rate,
  pressure_index, final_score, winner

Sampling strategy
  One snapshot row is emitted at the end of every 2nd completed over
  (overs 2, 4, 6 ... 20) for each innings.  This gives the model many
  partial-innings views of each match rather than just the end result.

Usage:
    python data/prepare_real_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(_HERE, "raw")
DEL_PATH    = os.path.join(RAW_DIR, "deliveries.csv")
MATCH_PATH  = os.path.join(RAW_DIR, "matches.csv")
OUT_PATH    = os.path.join(_HERE, "matches.csv")

TOTAL_OVERS = 20.0
SAMPLE_EVERY_N_OVERS = 2        # emit a snapshot every N completed overs


# -- Helpers -------------------------------------------------------------------

def _validate_inputs() -> None:
    for p in (DEL_PATH, MATCH_PATH):
        if not os.path.exists(p):
            sys.exit(
                f"[X] Missing: {p}\n"
                "    Download the Kaggle dataset first:\n"
                "    kaggle datasets download -d patrickb1912/ipl-complete-dataset-20082020 "
                "--path data/raw --unzip"
            )


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[*] Loading raw Kaggle CSVs ...")
    deliveries = pd.read_csv(DEL_PATH)
    matches    = pd.read_csv(MATCH_PATH)
    print(f"    deliveries : {deliveries.shape[0]:,} rows x {deliveries.shape[1]} cols")
    print(f"    matches    : {matches.shape[0]:,} rows x {matches.shape[1]} cols")
    return deliveries, matches


def _build_innings_totals(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Return each (match_id, inning) final score and wickets."""
    agg = (
        deliveries
        .groupby(["match_id", "inning"], sort=True)
        .agg(final_runs=("total_runs", "sum"), total_wickets=("is_wicket", "sum"))
        .reset_index()
    )
    return agg


def _build_snapshots(deliveries: pd.DataFrame, innings_totals: pd.DataFrame,
                     matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each (match_id, inning), iterate over completed overs and emit one
    snapshot row every SAMPLE_EVERY_N_OVERS overs.
    """
    # Build a match-level lookup: match_id -> {winner, team1, team2}
    match_info = matches.set_index("id")[["winner", "team1", "team2"]].to_dict("index")

    # Build an innings-level lookup for final score
    totals_idx = innings_totals.set_index(["match_id", "inning"])

    records = []

    grouped = deliveries.groupby(["match_id", "inning"], sort=True)

    for (match_id, inning), grp in grouped:
        grp = grp.sort_values(["over", "ball"])

        # Cumulative stats ball-by-ball
        grp = grp.copy()
        grp["cum_runs"]    = grp["total_runs"].cumsum()
        grp["cum_wickets"] = grp["is_wicket"].cumsum()

        # Final score for this innings
        try:
            final_runs = int(totals_idx.loc[(match_id, inning), "final_runs"])
        except KeyError:
            continue

        # For 2nd innings: get target from matches table
        if inning == 2:
            m = matches[matches["id"] == match_id]
            if m.empty or pd.isna(m.iloc[0]["target_runs"]):
                # No valid target -> skip this innings
                continue
            target = int(m.iloc[0]["target_runs"])
        else:
            target = 0

        winner = match_info.get(match_id, {}).get("winner", "Unknown")

        # Desired sample points: end of over 2, 4, 6, ... 20  (1-indexed overs completed)
        # "over" column is 0-indexed, so over=1 means 2 overs completed.
        last_over_in_innings = int(grp["over"].max())

        for target_overs_completed in range(SAMPLE_EVERY_N_OVERS, 21, SAMPLE_EVERY_N_OVERS):
            over_idx = target_overs_completed - 1   # 0-indexed over number

            # Take rows up to and including this over
            mask = grp["over"] <= over_idx
            if not mask.any():
                break   # innings was shorter than this checkpoint

            snapshot = grp[mask].iloc[-1]

            overs_completed = min(target_overs_completed, last_over_in_innings + 1)
            current_score   = int(snapshot["cum_runs"])
            wickets_fallen  = int(snapshot["cum_wickets"])

            if overs_completed <= 0:
                continue

            current_run_rate = round(current_score / overs_completed, 4)
            remaining_overs  = round(TOTAL_OVERS - overs_completed, 1)
            wickets_in_hand  = max(0, 10 - wickets_fallen)

            # 2nd-innings chase features
            if inning == 2 and target > 0:
                runs_required   = max(0, target - current_score)
                balls_remaining = int(remaining_overs * 6)
                required_run_rate = (
                    round(runs_required / remaining_overs, 2)
                    if remaining_overs > 0 else 99.0
                )
                pressure_index = (
                    round(required_run_rate / current_run_rate, 4)
                    if current_run_rate > 0 else 99.0
                )
            else:
                runs_required     = 0
                balls_remaining   = int(remaining_overs * 6)
                required_run_rate = 0.0
                pressure_index    = 0.0

            records.append({
                "innings":             inning,
                "batting_team":        snapshot["batting_team"],
                "bowling_team":        snapshot["bowling_team"],
                "overs_completed":     overs_completed,
                "wickets_fallen":      wickets_fallen,
                "current_score":       current_score,
                "current_run_rate":    current_run_rate,
                "remaining_overs":     remaining_overs,
                "wickets_in_hand":     wickets_in_hand,
                "target":              target,
                "runs_required":       runs_required,
                "balls_remaining":     balls_remaining,
                "required_run_rate":   required_run_rate,
                "pressure_index":      pressure_index,
                "final_score":         final_runs,
                "winner":              winner,
            })

    return pd.DataFrame(records)


def _sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same guards as preprocess.clean_data so we know the output is valid."""
    before = len(df)
    df = df.dropna(subset=[
        "batting_team", "bowling_team", "overs_completed",
        "wickets_fallen", "current_score", "current_run_rate", "final_score"
    ])
    df = df[df["overs_completed"] > 0]
    df = df[df["wickets_fallen"].between(0, 10)]
    df = df[df["current_score"] >= 0]
    df = df[df["final_score"] >= df["current_score"]]
    after = len(df)
    if before != after:
        print(f"[i] Sanity-check dropped {before - after} rows -> {after} remaining")
    return df.reset_index(drop=True)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    _validate_inputs()
    deliveries, matches = _load_raw()

    print("[*] Computing innings totals ...")
    innings_totals = _build_innings_totals(deliveries)

    print(f"[*] Building snapshots (every {SAMPLE_EVERY_N_OVERS} overs) ...")
    df = _build_snapshots(deliveries, innings_totals, matches)
    print(f"    Raw snapshots  : {len(df):,}")

    df = _sanity_check(df)

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n[OK] Saved -> {OUT_PATH}")
    print(f"    Shape          : {df.shape}")
    print(f"    1st-innings    : {(df['innings'] == 1).sum():,} rows")
    print(f"    2nd-innings    : {(df['innings'] == 2).sum():,} rows")
    print(f"    Teams          : {sorted(df['batting_team'].unique())}")
    print(f"    Overs sampled  : {sorted(df['overs_completed'].unique())}")
    print(f"    final_score    : min={df['final_score'].min()}  "
          f"max={df['final_score'].max()}  mean={df['final_score'].mean():.1f}")
    print()
    print(df.head(4).to_string())


if __name__ == "__main__":
    main()
