"""End to end against the running Compose stack. CI runs these after replaying games into Redpanda.

    E2E_START=2026-07-01 E2E_GAMES=30 pytest -m integration
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]
DSN = os.environ.get("DATABASE_URL", "postgresql://abs:abs@localhost:5432/abs")
API = os.environ.get("API_URL", "http://localhost:8000")
GRAFANA = os.environ.get("GRAFANA_URL", "http://localhost:3000")
PROM = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")


def get(url, **headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def query(sql, *args):
    import psycopg

    with psycopg.connect(DSN) as conn:
        cur = conn.execute(sql, args)
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


@pytest.fixture(scope="module")
def expected():
    from abslive.policy import Tables
    from abslive.registry import resolve
    from abslive.replay import replay_frame
    from stream.producer import load

    pitches = load(str(ROOT / "data" / "replay_2026.csv.gz"), os.environ.get("E2E_START", "2026-07-01"), None,
                   int(os.environ.get("E2E_GAMES", "30")))
    directory, version = resolve()
    return replay_frame(Tables.load(directory, version=version), pitches), version


def wait_for_rows(n, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        have = int(query("SELECT count(*) AS n FROM decisions").n[0])
        if have >= n:
            return have
        time.sleep(3)
    raise AssertionError(f"only {have} of {n} pitches reached Postgres")


def test_stream_matches_the_batch_replay_exactly(expected):
    want, version = expected
    wait_for_rows(len(want))
    got = query("SELECT * FROM decisions WHERE game_pk = ANY(%s)", list(map(int, want.game_pk.unique())))
    assert len(got) == len(want)
    key = ["game_pk", "at_bat_number", "pitch_number"]
    m = want.merge(got, on=key, suffixes=("", "_db"))
    assert len(m) == len(want)
    for col in ["tokens_before", "tokens_actual", "optimal_challenge", "ump_wrong", "actual_challenge",
                "actual_success", "chal_team", "inning"]:
        assert (m[col] == m[f"{col}_db"]).all(), col
    for col in ["w_in", "y_in", "p_miss", "g_wp", "lam", "breakeven_p", "optimal_gain_wp", "actual_gain_wp",
                "actual_cva_wp"]:
        assert np.allclose(m[col], m[f"{col}_db"], rtol=0, atol=1e-9), col
    assert (got.model_version == version).all() and version.startswith("abs-challenge-policy/v")


def test_api_serves_the_champion(expected):
    _, version = expected
    assert get(f"{API}/health")["model_version"] == version
    body = json.dumps(dict(inning=7, half="bottom", outs=1, balls=2, strikes=2, on_second=True, home_score=2,
                           away_score=3, side="batter", challenges_left=2, perceived_miss_in=-1.5)).encode()
    req = urllib.request.Request(f"{API}/advise", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        out = json.loads(r.read())
    assert out["recommendation"] == "challenge" and out["model_version"] == version


def test_grafana_and_prometheus_are_wired_up():
    auth = {"Authorization": "Basic " + __import__("base64").b64encode(b"admin:admin").decode()}
    found = get(f"{GRAFANA}/api/search?query=ABS", **auth)
    assert any(d["uid"] == "abs-live" for d in found)
    for uid in ("abs-pg", "abs-prom"):
        assert get(f"{GRAFANA}/api/datasources/uid/{uid}/health", **auth)["status"] == "OK", uid
    deadline = time.time() + 60
    while True:
        targets = get(f"{PROM}/api/v1/targets")["data"]["activeTargets"]
        health = {t["labels"]["job"]: t["health"] for t in targets}
        if all(h == "up" for h in health.values()) or time.time() > deadline:
            break
        time.sleep(3)
    assert health == {"api": "up", "stream": "up", "monitor": "up"}


def _alert():
    r = get(f"{PROM}/api/v1/query?query=abs_drift_alert")["data"]["result"]
    if not r or r[0]["value"][1] == "NaN":
        return None
    return int(float(r[0]["value"][1]))


def wait_for_alert(value, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _alert() == value:
            return
        time.sleep(3)
    raise AssertionError(f"abs_drift_alert never became {value} (last {_alert()})")


@pytest.mark.skipif(os.environ.get("E2E_PHASE") == "drift", reason="checks the quiet phase")
def test_monitor_is_quiet_on_real_games():
    deadline = time.time() + 120
    while query("SELECT count(*) AS n FROM drift_checks").n[0] == 0:
        assert time.time() < deadline, "the monitor never ran a check"
        time.sleep(3)
    wait_for_alert(0)
    checks = query("SELECT * FROM drift_checks ORDER BY checked_at DESC LIMIT 1")
    assert not checks.alert[0] and checks.psi_miss_batter[0] < 0.1


@pytest.mark.skipif(os.environ.get("E2E_PHASE") != "drift", reason="needs the synthetic drift replay")
def test_monitor_fires_on_synthetic_drift():
    wait_for_alert(1)
    assert query("SELECT bool_or(alert) AS a FROM drift_checks").a[0]
