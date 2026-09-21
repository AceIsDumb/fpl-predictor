"""Predict next-gameweek points for every player and print the top picks per position.

Run after fetch_data.py -> build_features.py -> train_model.py. Players with two
fixtures (double gameweek) have their fixture predictions summed. Injured, suspended
and unavailable players are dropped, as are those below --min-chance.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
POS_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def team_names():
    path = RAW / "bootstrap-static.json"
    if not path.exists():
        return {}
    return {t["id"]: t["short_name"] for t in json.loads(path.read_text()).get("teams", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=10, help="players shown per position")
    ap.add_argument("--min-chance", type=int, default=75,
                    help="drop players whose chance of playing is below this %% (flagged by the FPL API)")
    args = ap.parse_args()

    art = joblib.load(ROOT / "fpl_model.pkl")
    up = pd.read_csv(PROCESSED / "upcoming_features.csv")
    if up.empty:
        raise SystemExit("No upcoming fixtures found - run fetch_data.py and build_features.py first.")

    up = up[~up["status"].isin(["i", "s", "n"])].copy()  # injured, suspended, not available
    up["pred"] = art["model"].predict(up[art["features"]])
    up["chance"] = up["chance_of_playing_next_round"].fillna(100)  # blank means no fitness concern
    agg = up.groupby(["element", "name", "position", "team"], as_index=False).agg(
        gw=("round", "first"), fixtures=("fixture", "count"), price=("price", "first"),
        chance=("chance", "first"), pred_points=("pred", "sum"))
    agg["pos"] = agg["position"].map(POS_NAMES)
    agg["team_name"] = agg["team"].map(team_names()).fillna(agg["team"].astype(str))
    agg["pts_per_m"] = agg["pred_points"] / agg["price"]
    agg = agg[agg["chance"] >= args.min_chance].sort_values("pred_points", ascending=False)

    gw = int(agg["gw"].iloc[0])
    out = PROCESSED / f"predictions_gw{gw}.csv"
    agg.to_csv(out, index=False)
    cols = ["name", "team_name", "price", "fixtures", "pred_points", "pts_per_m"]
    for pos in POS_NAMES.values():
        print(f"\nGW{gw} top {args.top} {pos}")
        print(agg[agg["pos"] == pos].head(args.top)[cols].round(2).to_string(index=False))
    print(f"\nall predictions -> {out}")


if __name__ == "__main__":
    main()
