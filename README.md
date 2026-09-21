# FPL points model

Predicts a player's Fantasy Premier League points for upcoming fixtures.

## Setup
```bash
cd model
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (from the `model/` directory)
```bash
python src/weekly.py            # fetch -> features -> train -> predict -> best squad
python src/weekly.py --skip-fetch --model rf --budget 100.5    # options
```
Run it after each gameweek's results are in and before the next deadline. Or run the steps individually:
`fetch_data.py`, `build_features.py`, `train_model.py`, `predict_gameweek.py`, `select_squad.py`.

## Layout
- `src/fetch_data.py`: official FPL API (players, fixtures, per-player history, per-gameweek live stats) plus archived past seasons.
- `src/build_features.py`: rolling 3/5-match averages (points, xG, xA, xGI, minutes, BPS, ICT), FDR, team and opponent goal form.
- `src/train_model.py`: chronological train/test split, MAE vs two baselines, model saved with joblib.
- `src/predict_gameweek.py`: scores the next gameweek's fixtures; double gameweeks are summed, injured/suspended players dropped.
- `src/select_squad.py`: integer program picking the 15-man squad, XI, captain and bench under budget, 3-per-club and formation rules.
- `src/weekly.py`: runs all of the above in order.

## Known limitations
- The squad is built from scratch each week (wildcard-style); it does not yet account for your existing team, transfers or -4 hits.
- It optimises the next gameweek only, so week-to-week picks may churn.
- Rolling features reset each season (player ids change), so early-season predictions rely on few matches.
- Past seasons use the archive at github.com/vaastav/Fantasy-Premier-League; the official API only serves the current season per gameweek.
- The 2025/26 defensive-contribution scoring change means older seasons' points are not perfectly comparable.
