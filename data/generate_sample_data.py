"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic T20 dataset covering BOTH innings.

1st-innings rows: target = 0, winner determined after full match simulation.
2nd-innings rows: target > 0, winner determined by whether chasing team
                  reached the target.

Run:
    python data/generate_sample_data.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
    "Sunrisers Hyderabad", "Punjab Kings",
]

TOTAL_OVERS = 20.0
NUM_MATCHES = 1500   # produces ~3000 rows (one snapshot per innings per match)


def _simulate_final_score(crr: float, remaining_overs: float, wickets_in_hand: int) -> int:
    """Simulate what a team actually ends up scoring from a snapshot."""
    # Projected additional runs with some variance and wicket decay
    decay = 1 - (10 - wickets_in_hand) * 0.04
    projected_additional = remaining_overs * crr * decay * np.random.uniform(0.85, 1.15)
    return int(max(0, round(projected_additional)))


def generate_dataset(n_matches: int = NUM_MATCHES) -> pd.DataFrame:
    records = []

    for _ in range(n_matches):
        team_a, team_b = np.random.choice(TEAMS, size=2, replace=False)

        # ── 1st Innings ──────────────────────────────────────────────────────
        overs_1 = round(np.random.uniform(5.0, 19.5), 1)
        wickets_1 = np.random.randint(0, 10)
        crr_1 = max(3.0, round(np.random.uniform(5.5, 10.5) - wickets_1 * 0.12
                               + np.random.normal(0, 0.3), 2))
        current_1 = int(crr_1 * overs_1)
        remaining_1 = round(TOTAL_OVERS - overs_1, 1)
        wih_1 = 10 - wickets_1
        projected_1 = (df_1_additional := _simulate_final_score(crr_1, remaining_1, wih_1))
        final_1 = current_1 + projected_1

        # ── 2nd Innings ──────────────────────────────────────────────────────
        target = final_1 + 1                      # team_b needs this to win
        overs_2 = round(np.random.uniform(5.0, 19.5), 1)
        wickets_2 = np.random.randint(0, 10)
        crr_2 = max(3.0, round(np.random.uniform(5.0, 11.0) - wickets_2 * 0.12
                               + np.random.normal(0, 0.3), 2))
        current_2 = min(int(crr_2 * overs_2), target - 1)  # can't already have won
        remaining_2 = round(TOTAL_OVERS - overs_2, 1)
        wih_2 = 10 - wickets_2
        final_2 = current_2 + _simulate_final_score(crr_2, remaining_2, wih_2)

        # Chase outcome
        chasing_wins = int(final_2 >= target)
        winner_match = team_b if chasing_wins else team_a

        # 2nd-innings specific features
        runs_required = target - current_2
        balls_remaining = int(remaining_2 * 6)
        rrr = round(runs_required / remaining_2, 2) if remaining_2 > 0 else 99.0
        pressure_index = round(rrr / crr_2, 4) if crr_2 > 0 else 99.0

        # ── Append 1st-innings row ────────────────────────────────────────────
        records.append({
            "innings": 1,
            "batting_team": team_a,
            "bowling_team": team_b,
            "overs_completed": overs_1,
            "wickets_fallen": wickets_1,
            "current_score": current_1,
            "current_run_rate": crr_1,
            "remaining_overs": remaining_1,
            "wickets_in_hand": wih_1,
            "target": 0,
            "runs_required": 0,
            "balls_remaining": int(remaining_1 * 6),
            "required_run_rate": 0.0,
            "pressure_index": 0.0,
            "final_score": final_1,
            "winner": winner_match,    # overall match winner
        })

        # ── Append 2nd-innings row ────────────────────────────────────────────
        records.append({
            "innings": 2,
            "batting_team": team_b,
            "bowling_team": team_a,
            "overs_completed": overs_2,
            "wickets_fallen": wickets_2,
            "current_score": current_2,
            "current_run_rate": crr_2,
            "remaining_overs": remaining_2,
            "wickets_in_hand": wih_2,
            "target": target,
            "runs_required": max(0, runs_required),
            "balls_remaining": max(0, balls_remaining),
            "required_run_rate": rrr,
            "pressure_index": pressure_index,
            "final_score": final_2,
            "winner": winner_match,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matches.csv")
    df.to_csv(output_path, index=False)
    print(f"[✓] Dataset saved  → {output_path}")
    print(f"    Shape          : {df.shape}")
    print(f"    1st-innings    : {(df['innings'] == 1).sum()} rows")
    print(f"    2nd-innings    : {(df['innings'] == 2).sum()} rows")
    print(f"    Winner balance : {df['winner'].value_counts().to_dict()}")
    print()
    print(df.head(4).to_string())
