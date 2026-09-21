"""Turn raw FPL data into a model-ready feature table.

Reads  model/data/raw/  (current season JSON + archived past seasons)
Writes model/data/processed/
  training_features.csv   one row per played player-fixture (has the target)
  upcoming_features.csv   one row per player-fixture in the next gameweek
  feature_columns.json    feature names shared by train/predict

Every rolling feature is shifted by one match, so a row only "sees" matches
played before it (no leakage). "Last N gameweeks" means the last N matches the
player has a row for; DGW players get one row per fixture.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

POS_MAP = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}  # archive uses GK, API uses element_type
WINDOWS = (3, 5)
ROLL_STATS = [
    "total_points", "minutes", "starts", "goals_scored", "assists", "clean_sheets", "saves",
    "bonus", "bps", "ict_index", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
]
NUMERIC = ROLL_STATS + ["value"]
FIXTURE_COLS = ["id", "event", "kickoff_time", "team_h", "team_a", "team_h_score",
                "team_a_score", "finished", "team_h_difficulty", "team_a_difficulty"]


def season_key(season):
    return int(season[:4])


def to_bool(s):
    return s.astype(str).str.lower().eq("true")


def ensure_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return df


def clean_fixtures(fx, season):
    fx = fx[FIXTURE_COLS].copy()
    fx["finished"] = to_bool(fx["finished"])
    fx["season"] = season
    return fx


# ----------------------------------------------------------------- loading
def load_archive_season(season):
    d = RAW / "history" / season
    gw = pd.read_csv(d / "merged_gw.csv", low_memory=False)
    gw = gw[gw["position"].isin(POS_MAP)].copy()  # drops manager rows ("AM")
    gw["position"] = gw["position"].map(POS_MAP)
    gw = gw.drop_duplicates(["element", "fixture"])
    gw["season"] = season
    gw["was_home"] = to_bool(gw["was_home"])
    fx = clean_fixtures(pd.read_csv(d / "fixtures.csv"), season)
    return gw[["season", "element", "name", "position", "fixture", "round", "was_home"]
              + [c for c in NUMERIC if c in gw.columns]], fx


def load_current():
    """Current season from the official-API JSON. Returns (player rows, fixtures, upcoming rows) or None."""
    boot_path, fx_path = RAW / "bootstrap-static.json", RAW / "fixtures.json"
    if not (boot_path.exists() and fx_path.exists()):
        return None
    boot = json.loads(boot_path.read_text())
    year = int(boot["events"][0]["deadline_time"][:4])
    season = f"{year}-{str(year + 1)[2:]}"
    fixtures = clean_fixtures(pd.DataFrame(json.loads(fx_path.read_text())), season)
    players = pd.DataFrame(boot["elements"])

    rows = []
    for path in (RAW / "element-summary").glob("*.json"):
        rows.extend(json.loads(path.read_text()).get("history", []))
    hist = pd.DataFrame(rows)
    if hist.empty:
        hist = pd.DataFrame(columns=["element", "fixture", "round", "was_home"])
    hist = hist.merge(players[["id", "web_name", "element_type"]],
                      left_on="element", right_on="id", how="left")
    hist = hist.rename(columns={"web_name": "name", "element_type": "position"})
    hist["season"] = season
    hist["was_home"] = to_bool(hist["was_home"])
    hist = hist.drop_duplicates(["element", "fixture"])
    hist = ensure_numeric(hist, NUMERIC)[["season", "element", "name", "position", "fixture",
                                          "round", "was_home"] + NUMERIC]

    # Next gameweek's fixtures, one row per (player, fixture)
    nxt = next((e["id"] for e in boot["events"] if e.get("is_next")), None)
    if nxt is None:
        nxt = next((e["id"] for e in boot["events"] if e.get("is_current") and not e["finished"]), None)
    upcoming = pd.DataFrame()
    if nxt is not None:
        fx = fixtures[fixtures["event"] == nxt]
        sides = pd.concat([
            fx[["id", "event", "team_h"]].rename(columns={"team_h": "team"}).assign(was_home=True),
            fx[["id", "event", "team_a"]].rename(columns={"team_a": "team"}).assign(was_home=False),
        ]).rename(columns={"id": "fixture", "event": "round"})
        pl = players[players["status"] != "u"][["id", "web_name", "element_type", "team", "now_cost",
                                                "status", "chance_of_playing_next_round"]]
        pl = pl.rename(columns={"id": "element", "web_name": "name", "element_type": "position",
                                "now_cost": "value"})
        upcoming = pl.merge(sides, on="team").drop(columns="team")
        upcoming["season"] = season
        upcoming["is_upcoming"] = True
    return hist, fixtures, upcoming


# ---------------------------------------------------------------- features
def grouped_roll(df, keys, cols, window):
    """Mean of the previous `window` matches per group (current match excluded)."""
    shifted = df.groupby(keys, sort=False)[cols].shift(1)
    rolled = shifted.groupby([df[k] for k in keys], sort=False).rolling(window, min_periods=1).mean()
    return rolled.droplevel(list(range(len(keys)))).reindex(df.index)


def team_context(fixtures):
    """Per team-match rolling goals for/against, plus FDR from the team's own side."""
    f = fixtures.copy()
    f["kickoff_time"] = pd.to_datetime(f["kickoff_time"], utc=True, errors="coerce")
    for c in ["team_h_score", "team_a_score", "team_h_difficulty", "team_a_difficulty"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    played = f["finished"]
    common = ["season", "fixture", "team", "opp", "was_home", "fdr", "round", "kickoff_time", "gf", "ga"]
    home = pd.DataFrame({"season": f.season, "fixture": f.id, "team": f.team_h, "opp": f.team_a,
                         "was_home": True, "fdr": f.team_h_difficulty, "round": f.event,
                         "kickoff_time": f.kickoff_time,
                         "gf": f.team_h_score.where(played), "ga": f.team_a_score.where(played)})
    away = pd.DataFrame({"season": f.season, "fixture": f.id, "team": f.team_a, "opp": f.team_h,
                         "was_home": False, "fdr": f.team_a_difficulty, "round": f.event,
                         "kickoff_time": f.kickoff_time,
                         "gf": f.team_a_score.where(played), "ga": f.team_h_score.where(played)})
    t = pd.concat([home, away])[common]
    t = t.sort_values(["season", "team", "round", "kickoff_time", "fixture"]).reset_index(drop=True)
    roll = grouped_roll(t, ["season", "team"], ["gf", "ga"], 5)
    t["team_gf_5"], t["team_ga_5"] = roll["gf"], roll["ga"]
    return t


def build(seasons):
    parts, fixtures = [], []
    for s in seasons:
        gw, fx = load_archive_season(s)
        parts.append(gw)
        fixtures.append(fx)

    upcoming = pd.DataFrame()
    current = load_current()
    if current is not None:
        hist, fx, upcoming = current
        # the archive may already hold this season once it is finished
        fixtures = [f for f in fixtures if f["season"].iat[0] != fx["season"].iat[0]]
        fixtures.append(fx)
        if len(hist):
            parts = [p for p in parts if p["season"].iat[0] != hist["season"].iat[0]]
            parts.append(hist)
        print(f"current season {fx['season'].iat[0]}: {len(hist)} played rows, {len(upcoming)} upcoming rows")

    fixtures = pd.concat(fixtures, ignore_index=True)
    players = pd.concat(parts + ([upcoming] if len(upcoming) else []), ignore_index=True)
    players["is_upcoming"] = players.get("is_upcoming", False)
    players["is_upcoming"] = players["is_upcoming"].fillna(False).astype(bool)
    players = ensure_numeric(players, NUMERIC + ["position", "element", "fixture", "round"])
    players["season_key"] = players["season"].map(season_key)

    # one kickoff time source for every row
    kick = fixtures[["season", "id", "kickoff_time"]].rename(columns={"id": "fixture"})
    kick["kickoff_time"] = pd.to_datetime(kick["kickoff_time"], utc=True, errors="coerce")
    players = players.merge(kick, on=["season", "fixture"], how="left")
    players = players.sort_values(["season_key", "element", "round", "kickoff_time", "fixture"]
                                  ).reset_index(drop=True)

    # --- player rolling form (shifted: uses only earlier matches)
    keys = ["season", "element"]
    feats = {}
    for w in WINDOWS:
        roll = grouped_roll(players, keys, ROLL_STATS, w)
        for c in ROLL_STATS:
            feats[f"{c.replace('total_points', 'points')}_avg_{w}"] = roll[c]
    shifted_pts = players.groupby(keys, sort=False)["total_points"].shift(1)
    feats["points_ewm"] = (shifted_pts.groupby([players[k] for k in keys], sort=False)
                           .ewm(halflife=2, min_periods=1).mean()
                           .droplevel([0, 1]).reindex(players.index))
    for c, name in [("total_points", "points_season_avg"), ("minutes", "minutes_season_avg")]:
        sh = players.groupby(keys, sort=False)[c].shift(1)
        feats[name] = (sh.groupby([players[k] for k in keys], sort=False)
                       .expanding(min_periods=1).mean().droplevel([0, 1]).reindex(players.index))
    feats["prev_minutes"] = players.groupby(keys, sort=False)["minutes"].shift(1)
    feats["games_before"] = players.groupby(keys, sort=False).cumcount()
    players = pd.concat([players, pd.DataFrame(feats)], axis=1)

    # --- team / opponent context and FDR
    tt = team_context(fixtures)
    own = tt[["season", "fixture", "was_home", "team", "opp", "fdr", "team_gf_5", "team_ga_5"]]
    opp = (tt[["season", "fixture", "team", "team_gf_5", "team_ga_5"]]
           .rename(columns={"team": "opp", "team_gf_5": "opp_gf_5", "team_ga_5": "opp_ga_5"}))
    players = players.merge(own, on=["season", "fixture", "was_home"], how="left")
    players = players.merge(opp, on=["season", "fixture", "opp"], how="left")

    # --- static row features
    players["price"] = players["value"] / 10
    players["was_home"] = players["was_home"].astype(int)
    for pos, name in {1: "gk", 2: "def", 3: "mid", 4: "fwd"}.items():
        players[f"pos_{name}"] = (players["position"] == pos).astype(int)

    feature_cols = (list(feats) + ["price", "was_home", "fdr", "team_gf_5", "team_ga_5",
                                   "opp_gf_5", "opp_ga_5", "pos_gk", "pos_def", "pos_mid", "pos_fwd"])
    ids = ["season", "season_key", "element", "name", "position", "team", "opp", "round",
           "fixture", "kickoff_time", "minutes", "total_points"]
    extra = [c for c in ["status", "chance_of_playing_next_round"] if c in players.columns]
    train = players[~players["is_upcoming"] & players["total_points"].notna()]
    up = players[players["is_upcoming"]]
    return train[ids + feature_cols], up[ids + extra + feature_cols], feature_cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="*", help="archive seasons to use (default: all downloaded)")
    args = ap.parse_args()
    seasons = args.seasons or sorted(p.name for p in (RAW / "history").iterdir() if p.is_dir())
    print("archive seasons:", seasons)

    train, up, feature_cols = build(seasons)
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "training_features.csv", index=False)
    up.to_csv(OUT / "upcoming_features.csv", index=False)
    (OUT / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    print(f"training rows: {len(train):,} | upcoming rows: {len(up):,} | features: {len(feature_cols)}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
