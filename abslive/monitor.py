"""Watch the live decisions for drift and miscalibration, and export the numbers to Prometheus.

Every INTERVAL seconds it reads the most recent WINDOW scored pitches from
Postgres and computes, against the champion model's training window:

  abs_drift_psi{feature="miss",side=...}   PSI of the true miss distance (ABS gives the truth)
  abs_drift_psi{feature="stake"}           PSI of the stakes
  abs_calibration{metric="ece"|"brier"}    how well P(call is wrong) matches what ABS ruled
  abs_drift_alert                          1 when any PSI passes PSI_ALERT or ECE passes ECE_ALERT

Each check is also written to the drift_checks table, so Grafana keeps the history.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from . import fit

PSI_ALERT = float(os.environ.get("PSI_ALERT", "0.2"))
ECE_ALERT = float(os.environ.get("ECE_ALERT", "0.02"))
WINDOW = int(os.environ.get("MONITOR_WINDOW", "5000"))
INTERVAL = float(os.environ.get("MONITOR_INTERVAL", "15"))
MIN_ROWS = int(os.environ.get("MONITOR_MIN_ROWS", "500"))

log = logging.getLogger("abslive.monitor")


def check(window: pd.DataFrame, reference: dict, bins: int = 10) -> dict:
    """Drift and calibration for one window of scored pitches. Pure function, unit-tested."""
    out: dict = {"rows": int(len(window))}
    for side in ("batter", "fielder"):
        cur = window.loc[window.chal_side == side, "w_in"].to_numpy()
        out[f"psi_miss_{side}"] = fit.psi(reference[f"w_{side}"], cur, fit.MISS_EDGES) if len(cur) else 0.0
    out["psi_stake"] = fit.psi(reference["g"], window.g_wp.to_numpy(), fit.STAKE_EDGES)
    p = window.p_miss.to_numpy(float)
    truth = window.ump_wrong.to_numpy(float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out["ece"] = float(sum(abs(p[idx == b].mean() - truth[idx == b].mean()) * (idx == b).mean()
                           for b in range(bins) if (idx == b).any()))
    out["brier"] = float(np.mean((p - truth) ** 2))
    out["miss_rate"] = float(truth.mean())
    psis = [out["psi_miss_batter"], out["psi_miss_fielder"], out["psi_stake"]]
    out["alert"] = int(max(psis) > PSI_ALERT or out["ece"] > ECE_ALERT)
    return out


def main() -> None:  # pragma: no cover - exercised by the integration tests
    import psycopg
    from prometheus_client import Gauge, Info, start_http_server

    from .registry import resolve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    model_dir, version = resolve()
    ref_file = model_dir / "reference.npz"
    if not ref_file.exists():  # the research export has no training window; use the whole season
        ref_file = model_dir / "reference_prior.npz"
    z = np.load(ref_file)
    reference = {"w_batter": z["w_batter"], "w_fielder": z["w_fielder"], "g": z["g"]}

    psi_g = Gauge("abs_drift_psi", "Population stability index against the training window", ["feature", "side"])
    cal_g = Gauge("abs_calibration", "Calibration of P(call is wrong) on the live window", ["metric"])
    rate_g = Gauge("abs_live_miss_rate", "Share of called pitches ABS says the umpire missed, live window")
    rows_g = Gauge("abs_monitor_window_rows", "Scored pitches in the monitored window")
    alert_g = Gauge("abs_drift_alert", "1 when drift or miscalibration passes its threshold")
    alert_g.set(float("nan"))   # unknown until the first check, not "no alert"
    Info("abs_monitor_model", "Model the monitor compares against").info({"version": version})
    start_http_server(int(os.environ.get("MONITOR_PORT", "9108")))

    dsn = os.environ.get("DATABASE_URL", "postgresql://abs:abs@localhost:5432/abs")
    log.info("monitoring %s against %s, window %d", dsn.split("@")[-1], version, WINDOW)
    while True:
        try:
            with psycopg.connect(dsn, autocommit=True) as conn:
                window = pd.DataFrame(conn.execute(
                    "SELECT chal_side, w_in, g_wp, p_miss, ump_wrong FROM decisions "
                    "ORDER BY scored_at DESC, game_pk DESC, at_bat_number DESC, pitch_number DESC LIMIT %s",
                    (WINDOW,)).fetchall(), columns=["chal_side", "w_in", "g_wp", "p_miss", "ump_wrong"])
                rows_g.set(len(window))
                if len(window) >= MIN_ROWS:
                    r = check(window, reference)
                    psi_g.labels("miss", "batter").set(r["psi_miss_batter"])
                    psi_g.labels("miss", "fielder").set(r["psi_miss_fielder"])
                    psi_g.labels("stake", "all").set(r["psi_stake"])
                    cal_g.labels("ece").set(r["ece"])
                    cal_g.labels("brier").set(r["brier"])
                    rate_g.set(r["miss_rate"])
                    alert_g.set(r["alert"])
                    conn.execute(
                        "INSERT INTO drift_checks (model_version, rows, psi_miss_batter, psi_miss_fielder, psi_stake,"
                        " ece, brier, miss_rate, alert) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (version, r["rows"], r["psi_miss_batter"], r["psi_miss_fielder"], r["psi_stake"],
                         r["ece"], r["brier"], r["miss_rate"], bool(r["alert"])))
                    log.info("check %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
        except Exception as exc:  # keep watching through restarts of the database
            log.warning("check failed: %s", exc)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
