# ⚽ FIFA World Cup 2026 — Match Outcome Predictor

A machine learning system that predicts match results for the FIFA World Cup 2026, simulates the full tournament from group stage to final, and exposes an interactive Streamlit dashboard.

---

## Overview

This project uses historical international football match data combined with FIFA player statistics to build a supervised classification system capable of predicting the outcome (win / draw / loss) of any football match. The trained model is then used to predict all FIFA World Cup 2026 matches, covering the 12 groups, the Round of 32, Round of 16, Quarter-finals, Semi-finals, Third-place play-off, and the Final.


---

## Objectives

- Build and compare multiple classification models to predict football match outcomes.
- Select the best-performing model and use it to simulate the entire FIFA World Cup 2026 bracket.
- Provide an interactive dashboard allowing users to predict individual group-stage matches and explore the full knockout bracket.

---

## Key Features

- **Multi-model benchmarking** : seven classifiers trained and evaluated on the same pipeline (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM).
- **Symmetric match prediction** : each match is predicted twice (swapping home/away roles) and probabilities are averaged to remove positional bias.
- **Complete tournament simulation** : group stage standings, best-third qualification logic, Round of 32 bracket construction, and full knockout elimination rounds up to the Final.
- **Interactive Streamlit app** : group prediction tool with probability bar charts, full knockout bracket viewer with per-stage Plotly visualisations, and a tournament podium summary.

---

## Project Structure

```
FIFA-World-Cup-2026-Outcomes-Prediction/
│
├── data/
│   └── teams_match_features.csv      # Preprocessed match dataset with ELO & FIFA stats
│
├── pictures/
│   ├── wc-logo.png                   # World Cup logo (used in Streamlit header)
│   └── pic2.png                      # Secondary header image
│
├── .streamlit/
│   └── config.toml                   # Streamlit theme configuration
│
├── world-cup.ipynb                   # Main Jupyter notebook (EDA, training, simulation)
├── app.py                            # Streamlit web application
└── README.md
```

---

## Dataset

**File:** `data/teams_match_features.csv`

The dataset covers international football matches from **1872 to the present**, enriched with:

| Feature group | Description |
|---|---|
| Match metadata | Tournament name, date, home team, away team, goals scored |
| Recent form | Average goals scored/conceded and win rate over the last 5 matches |
| FIFA player ratings | Average and maximum overall ratings per team; attack, defense, pace, shooting, passing sub-ratings |
| ELO ratings | Home ELO, away ELO, and their difference at the time of the match |


---


## Machine Learning Models

All models are wrapped in a `sklearn.pipeline.Pipeline` that applies `StandardScaler` before fitting, ensuring consistent preprocessing across classifiers.

| Model |
|---|
| Logistic Regression |
| K-Nearest Neighbours |
| Decision Tree |
| Random Forest | 
| Gradient Boosting |
| **XGBoost** ✅ |
| LightGBM

**XGBoost was selected as the best model** based on weighted F1-score and is used for all tournament simulation steps.

---

## Tournament Simulation

### Group Stage (12 groups × 4 teams)

- Every pair within a group plays once (6 matches per group, 72 total).
- Points are awarded: 3 for a win, 1 for a draw, 0 for a loss.
- The top 2 teams per group qualify directly; the 8 best third-placed teams also advance (48 qualified teams total, consistent with the 2026 expanded format).

### Knockout Rounds

| Round | Matches |
|---|---|
| Round of 32 | 16 matches (M73–M88) |
| Round of 16 | 8 matches (M89–M96) |
| Quarter-finals | 4 matches (M97–M100) |
| Semi-finals | 2 matches (M101–M102) |
| Third-place play-off | 1 match (M103) |
| **Final** | **1 match (M104)** |

In knockout rounds, draws are resolved by selecting the team with the higher predicted win probability (no penalty shootout modelling). Bracket construction follows the official FIFA 2026 slot assignment rules for best third-placed teams.


---

## Streamlit Dashboard

The `app.py` file implements a two-page Streamlit application:

**Page 1 — Group Prediction**
- Select any group and pick two teams to simulate a head-to-head match.
- Displays win/draw/loss probabilities as metric cards and an interactive Plotly bar chart.

**Page 2 — Knockout Bracket**
- Shows the full simulated tournament podium (Champion, Runner-up, 3rd, 4th).
- Multi-select filter to view results by stage (Round of 32 through Final).
- Grouped Plotly bar charts per stage showing match-by-match win probabilities.
- Expandable results table with formatted probability columns.

---

## Technologies & Libraries

| Category | Library / Tool |
|---|---|
| Language | Python  |
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Visualisation (notebook) | `matplotlib`, `seaborn` |
| Visualisation (app) | `plotly` |
| Web application | `streamlit`, `streamlit-extras` |

---

## Installation


### Steps

1. **Clone the repository**.

2. **Install dependencies:**

   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly streamlit streamlit-extras
   ```

---

### Streamlit Application

From the project root directory:

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## Cloning from GitHub

```bash
git clone https://github.com/mohamederrahmouni/FIFA-World-Cup-2026-Outcomes-Prediction.git
cd FIFA-World-Cup-2026-Outcomes-Prediction
```

---

## Author

**Mohamed Errahmouni**
Big Data & Artificial Intelligence Engineering Student

- GitHub: [@mohamederrahmouni](https://github.com/mohamederrahmouni)
- Repository: [FIFA-World-Cup-2026-Outcomes-Prediction](https://github.com/mohamederrahmouni/FIFA-World-Cup-2026-Outcomes-Prediction)

---

