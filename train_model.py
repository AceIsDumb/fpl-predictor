"""Train a regressor that predicts a player's FPL points in a fixture.

Split is chronological (the most recent gameweeks are the test set), because a
random split would let the model train on the future. Reports MAE against two
naive baselines, plus the average actual points of the model's top-15 picks per
gameweek (what matters for squad selection; MAE is dominated by zero-point rows).

Saves model/fpl_model.pkl (model + feature list) and model/metrics.json.
The saved model is refit on ALL data after evaluation.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def make_model(kind):
    if kind == "rf":  # random forests can't take NaNs, so impute first
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5,
                                  n_jobs=-1, random_state=42),
        )
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=50, l2_regularization=1.0, random_state=42)


def chronological_split(df, test_frac):
    gws = (df[["season_key", "round"]].drop_duplicates()
           .sort_values(["season_key", "round"]).reset_index(drop=True))
    cut = gws.iloc[int(len(gws) * (1 - test_frac))]
    is_test = (df["season_key"] > cut["season_key"]) | (
        (df["season_key"] == cut["season_key"]) & (df["round"] >= cut["round"]))
    return df[~is_test], df[is_test]


def top_n_points(test, scores, n=15):
    """Average actual points of the n highest-scored rows in each gameweek."""
    d = test[["season", "round", "total_points"]].assign(score=np.asarray(scores))
    return float(np.mean([g.nlargest(n, "score")["total_points"].mean()
                          for _, g in d.groupby(["season", "round"])]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["hgb", "rf"], default="hgb",
                    help="hgb = gradient boosting (default), rf = random forest")
    ap.add_argument("--test-frac", type=float, default=0.2, help="share of most recent gameweeks held out")
    args = ap.parse_args()

    df = pd.read_csv(PROCESSED / "training_features.csv")
    features = json.loads((PROCESSED / "feature_columns.json").read_text())
    train, test = chronological_split(df, args.test_frac)
    print(f"train: {len(train):,} rows | test: {len(test):,} rows "
          f"(from {test['season'].iloc[0]} GW{int(test['round'].min())} onward)")

    model = make_model(args.model).fit(train[features], train["total_points"])
    pred = model.predict(test[features])
    y = test["total_points"]

    baselines = {
        "train-mean": np.full(len(test), train["total_points"].mean()),
        "last-5-avg": test["points_avg_5"].fillna(train["total_points"].mean()).to_numpy(),
    }
    metrics = {"model": args.model, "mae": float(mean_absolute_error(y, pred)),
               "top15_avg_points": top_n_points(test, pred)}
    for name, b in baselines.items():
        metrics[f"baseline_{name}_mae"] = float(mean_absolute_error(y, b))
        metrics[f"baseline_{name}_top15_avg_points"] = top_n_points(test, b)
    print(json.dumps(metrics, indent=2))

    final = make_model(args.model).fit(df[features], df["total_points"])
    joblib.dump({"model": final, "features": features, "model_type": args.model}, ROOT / "fpl_model.pkl")
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {ROOT / 'fpl_model.pkl'}")


if __name__ == "__main__":
    main()
