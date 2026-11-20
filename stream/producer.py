"""Replay the 2026 season into Redpanda (Kafka API), as if the games were live.

    python -m stream.producer --start 2026-07-01 --days 3 --rate 500

Games on the same day are interleaved: the producer steps through every game
at the same pace, one pitch each in turn, so a dashboard sees a full slate in
progress at once. Each message is one called pitch, keyed by game so that a
game's pitches stay in order on one partition.

--drift-shift adds that many inches to the true miss distance of every pitch
from --drift-from on. It is synthetic, to show the drift monitor firing: the
real 2026 season barely drifts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOPIC = os.environ.get("PITCH_TOPIC", "pitches")


def interleave(day: pd.DataFrame) -> pd.DataFrame:
    """Order one day's pitches so all its games advance together."""
    day = day.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    day["seq"] = day.groupby("game_pk").cumcount()
    day["frac"] = day.seq / day.groupby("game_pk").seq.transform("max").clip(lower=1)
    return day.sort_values(["frac", "game_pk"]).drop(columns=["seq", "frac"])


def load(path: str, start: str, days: int | None, games: int | None) -> pd.DataFrame:
    season = pd.read_csv(path)
    season = season[season.game_date >= start]
    dates = sorted(season.game_date.unique())
    if days:
        season = season[season.game_date.isin(dates[:days])]
    if games:
        keep = season.drop_duplicates("game_pk").game_pk.iloc[:games]
        season = season[season.game_pk.isin(keep)]
    return season


def ensure_topic(bootstrap: str, partitions: int) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    if TOPIC not in admin.list_topics(timeout=10).topics:
        for f in admin.create_topics([NewTopic(TOPIC, num_partitions=partitions, replication_factor=1)]).values():
            try:
                f.result()
            except Exception as exc:  # created by another producer meanwhile
                if "TOPIC_ALREADY_EXISTS" not in str(exc):
                    raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "replay_2026.csv.gz"))
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092"))
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--games", type=int, default=None, help="stop after this many games")
    ap.add_argument("--rate", type=float, default=300, help="pitches per second; 0 = as fast as possible")
    ap.add_argument("--partitions", type=int, default=6)
    ap.add_argument("--drift-from", default=None, help="date from which to shift the true miss distance")
    ap.add_argument("--drift-shift", type=float, default=0.0, help="inches added to the miss distance")
    args = ap.parse_args(argv)

    from confluent_kafka import Producer

    season = load(args.data, args.start, args.days, args.games)
    if args.drift_from and args.drift_shift:
        m = season.game_date >= args.drift_from
        season.loc[m, "w_in"] = season.loc[m, "w_in"] + args.drift_shift
    ensure_topic(args.bootstrap, args.partitions)
    producer = Producer({"bootstrap.servers": args.bootstrap, "linger.ms": 20, "enable.idempotence": True})

    sent = 0
    started = time.time()
    for date, day in season.groupby("game_date", sort=True):
        ordered = interleave(day)
        ordered = ordered.astype(object).where(ordered.notna(), None)   # NaN is not valid JSON
        for rec in ordered.to_dict(orient="records"):
            producer.produce(TOPIC, key=str(rec["game_pk"]), value=json.dumps(rec, default=str))
            sent += 1
            if sent % 1000 == 0:
                producer.poll(0)
            if args.rate:
                behind = sent / args.rate - (time.time() - started)
                if behind > 0:
                    time.sleep(behind)
        print(f"{date}: {sent:,} pitches sent", flush=True)
    producer.flush(30)
    print(json.dumps({"sent": sent, "games": int(season.game_pk.nunique()), "seconds": round(time.time() - started, 1)}))
    return sent


if __name__ == "__main__":
    main()
