# 🏏 Cricket Score Prediction System

A machine-learning system that predicts the **final score** of a batting team
during an ongoing T20 match, based on current match conditions.

---

## Project Structure

```
cricket-score-predictor/
│
├── data/
│   ├── matches.csv                  ← training dataset (auto-generated if missing)
│   └── generate_sample_data.py      ← generates a realistic synthetic dataset
│
├── src/
│   ├── preprocess.py                ← data loading, cleaning, team encoding
│   ├── features.py                  ← feature engineering (remaining overs, etc.)
│   ├── train.py                     ← model training & evaluation pipeline
│   ├── predict.py                   ← inference / prediction logic
│   └── app.py                       ← (Bonus) Flask REST API
│
├── model/
│   ├── model.pkl                    ← saved best model (created after training)
│   ├── encoders.pkl                 ← saved LabelEncoders for team names
│   ├── feature_importance.png       ← RF feature importance chart
│   └── predictions_vs_actual.png    ← predicted vs actual scatter plot
│
├── main.py                          ← CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the System

### Option A – CLI (train + predict in one shot)

```bash
# First run: auto-generates data and trains the model, then launches the predictor
python main.py

# Force a retrain (e.g. after replacing matches.csv)
python main.py --train
```

### Option B – Run steps separately

```bash
# Step 1: generate sample data (skip if you have a real matches.csv)
python data/generate_sample_data.py

# Step 2: train the models
python src/train.py

# Step 3: run the predictor
python main.py
```

### Option C – Flask API (Bonus)

```bash
python src/app.py
```

Then call the endpoint:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"batting_team\": \"Mumbai Indians\", \"bowling_team\": \"Chennai Super Kings\", \"current_score\": 98, \"overs_completed\": 12.0, \"wickets_fallen\": 3, \"current_run_rate\": 8.17}"
```

---

## Input Parameters

| Field              | Type  | Description                              |
|--------------------|-------|------------------------------------------|
| `batting_team`     | str   | Name of the batting team                 |
| `bowling_team`     | str   | Name of the fielding team                |
| `current_score`    | int   | Runs scored so far                       |
| `overs_completed`  | float | Overs bowled (e.g. `12.3`)              |
| `wickets_fallen`   | int   | Wickets lost (0–10)                      |
| `current_run_rate` | float | CRR = current_score / overs_completed    |

---

## Engineered Features

| Feature              | Meaning                                                    |
|----------------------|------------------------------------------------------------|
| `remaining_overs`    | Overs left to bowl (20 − overs_completed)                  |
| `wickets_in_hand`    | Batting resources remaining (10 − wickets_fallen)          |
| `projected_score`    | Naive CRR-based final score projection                     |
| `scoring_rate_factor`| CRR normalised to average T20 run rate (≈ 7.5)            |

---

## Models Compared

| Model                     | Metric | Typical Value     |
|---------------------------|--------|-------------------|
| Linear Regression         | MAE    | ~12–15 runs       |
| Random Forest Regressor   | MAE    | ~7–10 runs        |

The model with the lower MAE is automatically saved as `model/model.pkl`.

---

## Expected Output (CLI)

```
╔══════════════════════════════════════════════════════╗
║                  PREDICTION RESULT                   ║
╠══════════════════════════════════════════════════════╣
║  Batting Team    : Mumbai Indians                    ║
║  Current Score   : 98                                ║
║  Overs Completed : 12.0                              ║
║  Wickets Fallen  : 3                                 ║
║  Remaining Overs : 8.0                               ║
║  Wickets in Hand : 7                                 ║
╠══════════════════════════════════════════════════════╣
║  🎯 Predicted Final Score : 178                      ║
║  📊 Confidence Range      : 168 – 188                ║
║  📈 Simple CRR Projection : 163                      ║
╚══════════════════════════════════════════════════════╝
```

---

## Tech Stack

- **Python 3.12**
- `pandas` · `numpy` – data handling
- `scikit-learn` – ML models (LinearRegression, RandomForestRegressor)
- `matplotlib` · `seaborn` – visualisations
- `flask` – REST API (bonus)
- `pickle` – model persistence

---

## Using Real IPL Data

1. Download an IPL ball-by-ball dataset (e.g., from [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)).
2. Pre-process it to match the required columns and save as `data/matches.csv`.
3. Run `python main.py --train` to retrain on real data.
