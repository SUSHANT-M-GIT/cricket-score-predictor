"""
app.py  (Bonus – Flask REST API)
---------------------------------
Two endpoints:
  POST /predict/score   ->  regression  (1st or 2nd innings)
  POST /predict/winner  ->  classification + score (2nd innings)

Run:
    python src/app.py

Example – score only (1st innings):
    POST /predict/score
    {
        "batting_team":    "Mumbai Indians",
        "bowling_team":    "Chennai Super Kings",
        "current_score":   98,
        "overs_completed": 12.0,
        "wickets_fallen":  3,
        "current_run_rate": 8.17
    }

Example – score + winner (2nd innings):
    POST /predict/winner
    {
        "batting_team":    "Chennai Super Kings",
        "bowling_team":    "Mumbai Indians",
        "current_score":   87,
        "overs_completed": 10.0,
        "wickets_fallen":  2,
        "current_run_rate": 8.7,
        "target":          175
    }
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from predict import predict_score, predict_winner, predict_full

app = Flask(__name__)


def _parse_body(required_fields: list[str]) -> tuple[dict | None, object]:
    data = request.get_json(silent=True)
    if not data:
        return None, jsonify({"error": "Request body must be JSON."}), 400
    missing = [f for f in required_fields if f not in data]
    if missing:
        return None, jsonify({"error": f"Missing fields: {missing}"}), 422
    return data, None


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Cricket Score + Winner Predictor",
        "endpoints": {
            "POST /predict/score":  "Predict final score (any innings)",
            "POST /predict/winner": "Predict score + winner (2nd innings)",
        },
    })


@app.route("/predict/score", methods=["POST"])
def api_predict_score():
    required = ["batting_team", "bowling_team", "current_score",
                "overs_completed", "wickets_fallen", "current_run_rate"]
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 422

    try:
        result = predict_score(
            current_score=int(data["current_score"]),
            overs_completed=float(data["overs_completed"]),
            wickets_fallen=int(data["wickets_fallen"]),
            batting_team=str(data["batting_team"]),
            bowling_team=str(data["bowling_team"]),
            current_run_rate=float(data["current_run_rate"]),
        )
        return jsonify({"input": data, "prediction": result})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/winner", methods=["POST"])
def api_predict_winner():
    required = ["batting_team", "bowling_team", "current_score",
                "overs_completed", "wickets_fallen", "current_run_rate", "target"]
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 422

    try:
        result = predict_full(
            current_score=int(data["current_score"]),
            overs_completed=float(data["overs_completed"]),
            wickets_fallen=int(data["wickets_fallen"]),
            batting_team=str(data["batting_team"]),
            bowling_team=str(data["bowling_team"]),
            current_run_rate=float(data["current_run_rate"]),
            target=int(data["target"]),
        )
        return jsonify({"input": data, "prediction": result})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
