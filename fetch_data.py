"""Fetch raw FPL data (JSON) into model/data/raw/.

Official FPL API endpoints used:
  bootstrap-static/        players, teams, gameweek calendar
  fixtures/                every fixture of the current season (with FDR)
  element-summary/{id}/    per-player fixture list + per-gameweek history
  event/{gw}/live/         all-player stats for one gameweek

The official API only exposes the *current* season per gameweek, so past seasons
(needed for training data) are downloaded from the community archive
github.com/vaastav/Fantasy-Premier-League (same field names as the API).
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fantasy.premierleague.com/api"
ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
HISTORY_DEFAULT = ["2022-23", "2023-24", "2024-25", "2025-26"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (fpl-model data fetcher)"})


def get(url, retries=4, pause=2.0):
    """GET with retries; the FPL API returns 503 while it is updating scores."""
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException:
            if attempt == retries:
                raise
        else:
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
                r.raise_for_status()
        time.sleep(pause * attempt)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fetch_core():
    boot = get(f"{BASE}/bootstrap-static/").json()
    save_json(boot, RAW_DIR / "bootstrap-static.json")
    pd.DataFrame(boot["elements"]).to_csv(DATA_DIR / "raw_players.csv", index=False)

    fixtures = get(f"{BASE}/fixtures/").json()
    save_json(fixtures, RAW_DIR / "fixtures.json")
    print(f"bootstrap-static: {len(boot['elements'])} players, {len(boot['events'])} gameweeks")
    print(f"fixtures: {len(fixtures)}")
    return boot


def fetch_player_summaries(boot, delay, resume):
    out_dir = RAW_DIR / "element-summary"
    ids = [p["id"] for p in boot["elements"]]
    failed = []
    for i, pid in enumerate(ids, 1):
        path = out_dir / f"{pid}.json"
        if resume and path.exists():
            continue
        try:
            save_json(get(f"{BASE}/element-summary/{pid}/").json(), path)
        except requests.RequestException as exc:
            failed.append(pid)
            print(f"  player {pid} failed: {exc}")
        time.sleep(delay)
        if i % 100 == 0:
            print(f"  element-summary {i}/{len(ids)}")
    print(f"element-summary: {len(ids) - len(failed)}/{len(ids)} saved")
    if failed:
        print(f"  re-run with --resume to retry: {failed}")


def fetch_live(boot, delay):
    out_dir = RAW_DIR / "event-live"
    n = 0
    for ev in boot["events"]:
        if not (ev["finished"] or ev.get("is_current")):
            continue
        path = out_dir / f"{ev['id']}.json"
        if path.exists() and ev.get("data_checked"):
            continue  # final numbers already stored
        save_json(get(f"{BASE}/event/{ev['id']}/live/").json(), path)
        n += 1
        time.sleep(delay)
    print(f"event-live: {n} gameweeks fetched")


def fetch_history(seasons, refresh):
    for season in seasons:
        for name, dest in [("gws/merged_gw.csv", "merged_gw.csv"), ("fixtures.csv", "fixtures.csv")]:
            path = RAW_DIR / "history" / season / dest
            if path.exists() and not refresh:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(get(f"{ARCHIVE}/{season}/{name}").content)
        print(f"history {season}: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="*", default=HISTORY_DEFAULT,
                    help="past seasons to download from the archive (pass none to skip)")
    ap.add_argument("--skip-players", action="store_true", help="skip the ~800 per-player requests")
    ap.add_argument("--resume", action="store_true", help="skip players already saved")
    ap.add_argument("--refresh-history", action="store_true", help="re-download archive files")
    ap.add_argument("--delay", type=float, default=0.15, help="seconds between API calls")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    boot = fetch_core()
    if not args.skip_players:
        fetch_player_summaries(boot, args.delay, args.resume)
    fetch_live(boot, args.delay)
    fetch_history(args.seasons, args.refresh_history)
    print(f"done -> {DATA_DIR}")


if __name__ == "__main__":
    main()
