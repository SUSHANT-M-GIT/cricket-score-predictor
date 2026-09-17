"""
main.py
-------
CLI entry point for the Cricket Score + Winner Prediction System.

Usage:
    python main.py            ->  interactive prediction
    python main.py --train    ->  (re)train both models first, then predict
    python main.py --help     ->  usage info
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

_ROOT = os.path.dirname(__file__)


# -- CLI helpers ---------------------------------------------------------------

def _banner() -> None:
    print()
    print("+==========================================================+")
    print("|   🏏  CRICKET SCORE + WINNER PREDICTION SYSTEM  🏏       |")
    print("|          T20 Match · Regression + Classification         |")
    print("+==========================================================+")
    print()


def _get_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        try:
            val = float(input(f"  {prompt}: ").strip())
            if lo <= val <= hi:
                return val
            print(f"    ⚠  Enter a value between {lo} and {hi}.")
        except ValueError:
            print("    ⚠  Invalid — enter a number.")


def _get_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            val = int(input(f"  {prompt}: ").strip())
            if lo <= val <= hi:
                return val
            print(f"    ⚠  Enter an integer between {lo} and {hi}.")
        except ValueError:
            print("    ⚠  Invalid — enter a whole number.")


def _get_team(prompt: str, exclude: str | None = None) -> str:
    while True:
        val = input(f"  {prompt}: ").strip()
        if not val:
            print("    ⚠  Team name cannot be empty.")
        elif val == exclude:
            print("    ⚠  Batting and bowling teams must be different.")
        else:
            return val


def _get_crr(current_score: int, overs_completed: float) -> float:
    auto = round(current_score / overs_completed, 2)
    print(f"\n  Auto-calculated Current Run Rate : {auto}")
    override = input("  Press Enter to accept, or type a value to override: ").strip()
    return float(override) if override else auto


# -- 1st innings output --------------------------------------------------------

def _print_score_result(batting_team: str, current_score: int,
                        overs_completed: float, wickets_fallen: int,
                        result: dict) -> None:
    W = 60
    print()
    print("+" + "=" * W + "+")
    print("|" + "  SCORE PREDICTION RESULT".center(W) + "|")
    print("+" + "=" * W + "+")

    def row(label: str, value) -> None:
        line = f"  {label:<28}: {str(value)}"
        print("|" + line.ljust(W) + "|")

    row("Batting Team", batting_team)
    row("Current Score", current_score)
    row("Overs Completed", overs_completed)
    row("Wickets Fallen", wickets_fallen)
    row("Remaining Overs", result["remaining_overs"])
    row("Wickets in Hand", result["wickets_in_hand"])
    print("+" + "=" * W + "+")
    row("🎯 Predicted Final Score", result["predicted_score"])
    conf = f"{result['lower_bound']} – {result['upper_bound']}"
    row("📊 Confidence Range", conf)
    row("📈 Simple CRR Projection", result["projected_score_crr"])
    print("+" + "=" * W + "+")
    print()


# -- 2nd innings output --------------------------------------------------------

def _print_full_result(batting_team: str, bowling_team: str,
                       current_score: int, overs_completed: float,
                       wickets_fallen: int, target: int,
                       result: dict) -> None:
    W = 60
    print()
    print("+" + "=" * W + "+")
    print("|" + "  FULL PREDICTION RESULT (2nd Innings)".center(W) + "|")
    print("+" + "=" * W + "+")

    def row(label: str, value) -> None:
        line = f"  {label:<28}: {str(value)}"
        print("|" + line.ljust(W) + "|")

    row("Batting Team", batting_team)
    row("Bowling Team", bowling_team)
    row("Target to Chase", target)
    row("Current Score", current_score)
    row("Overs Completed", overs_completed)
    row("Wickets Fallen", wickets_fallen)
    row("Remaining Overs", result["remaining_overs"])
    row("Wickets in Hand", result["wickets_in_hand"])
    print("+" + "=" * W + "+")
    row("Runs Required", result["runs_required"])
    row("Balls Remaining", result["balls_remaining"])
    row("Required Run Rate", f"{result['required_run_rate']:.2f}")
    row("Pressure Index", f"{result['pressure_index']:.2f}  (>1 = under pressure)")
    print("+" + "=" * W + "+")
    row("🎯 Predicted Final Score", result["predicted_score"])
    conf = f"{result['lower_bound']} – {result['upper_bound']}"
    row("📊 Confidence Range", conf)
    row("📈 Simple CRR Projection", result["projected_score_crr"])
    print("+" + "=" * W + "+")
    row("🏆 Predicted Winner", result["winner"])
    row(f"📈 {batting_team[:20]} Win%", f"{result['batting_team_prob']}%")
    row(f"📉 {bowling_team[:20]} Win%", f"{result['bowling_team_prob']}%")
    print("+" + "=" * W + "+")
    print()


# -- Main prediction flow ------------------------------------------------------

def run_prediction() -> None:
    from predict import predict_score, predict_full

    _banner()

    # Innings selection
    print("  Which innings is in progress?")
    print("    [1]  1st innings  (predict final score only)")
    print("    [2]  2nd innings  (predict score + winner)")
    while True:
        choice = input("\n  Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            innings = int(choice)
            break
        print("    ⚠  Please enter 1 or 2.")

    print()
    batting_team  = _get_team("Batting team name")
    bowling_team  = _get_team("Bowling team name", exclude=batting_team)
    current_score = _get_int("Current score (runs)", 0, 300)
    overs         = _get_float("Overs completed (e.g. 12.3)", 0.1, 19.5)
    wickets       = _get_int("Wickets fallen (0–10)", 0, 10)
    crr           = _get_crr(current_score, overs)

    print("\n  Predicting ...", end=" ", flush=True)

    try:
        if innings == 1:
            result = predict_score(
                current_score=current_score,
                overs_completed=overs,
                wickets_fallen=wickets,
                batting_team=batting_team,
                bowling_team=bowling_team,
                current_run_rate=crr,
            )
            print("done!")
            _print_score_result(batting_team, current_score, overs, wickets, result)

        else:
            target = _get_int("\n  Target score to chase", current_score + 1, 500)
            result = predict_full(
                current_score=current_score,
                overs_completed=overs,
                wickets_fallen=wickets,
                batting_team=batting_team,
                bowling_team=bowling_team,
                current_run_rate=crr,
                target=target,
            )
            print("done!")
            _print_full_result(batting_team, bowling_team,
                               current_score, overs, wickets, target, result)

    except FileNotFoundError as exc:
        print(f"\n\n  [ERROR] {exc}")
        sys.exit(1)

    again = input("  Run another prediction? (y/n): ").strip().lower()
    if again == "y":
        run_prediction()


# -- Training orchestrator -----------------------------------------------------

def run_training() -> None:
    import subprocess

    data_path = os.path.join(_ROOT, "data", "matches.csv")
    if not os.path.exists(data_path):
        print("[i] matches.csv not found — generating sample data ...")
        subprocess.run(
            [sys.executable,
             os.path.join(_ROOT, "data", "generate_sample_data.py")],
            check=True,
        )

    from train_regression     import train as train_reg
    from train_classification import train as train_clf

    train_reg(csv_path=data_path)
    print()
    train_clf(csv_path=data_path)


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cricket Score + Winner Predictor – T20 ML System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--train", action="store_true",
        help="(Re)train regression + classification models before predicting.",
    )
    args = parser.parse_args()

    if args.train:
        run_training()

    # Auto-train if models are missing
    reg_missing = not os.path.exists(os.path.join(_ROOT, "model", "model.pkl"))
    clf_missing = not os.path.exists(os.path.join(_ROOT, "model", "classifier.pkl"))

    if reg_missing or clf_missing:
        print("[i] One or more models missing — running full training pipeline ...\n")
        run_training()

    run_prediction()


if __name__ == "__main__":
    main()
