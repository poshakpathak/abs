"""Export the serving model and the replay data from the ABSGTO research project.

Run it from ABSGTO's src/ folder after `make base`:

    cd ABSGTO/src && python3 /path/to/abs-live/export/export_from_absgto.py /path/to/abs-live

It writes:

  model/research_tables.npz   everything the decision rule needs, taken from the solved paper model:
                              stake table g, shadow prices lam / lamx, posterior tables P(miss | y, sigma)
                              by challenging side, the grids they live on, and league-average sigma
  model/research_meta.json    provenance and the checks below
  data/replay_2026.csv.gz     every 2026 called pitch, in game order, with what the stream needs

Before writing, it checks the export against the paper's own outputs and stops if any differ:
  * shadow prices equal outputs/shadow_prices.csv
  * the stake table reproduces g_wp for every called pitch it covers
  * the posterior tables are monotone in y and lie in [0, 1]
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path.cwd()))  # run from ABSGTO/src
from abs_core import OUT  # noqa: E402

op = importlib.import_module("05_optimal_policy")
adv = importlib.import_module("37_export_advisor")

REPLAY_COLUMNS = [
    "game_pk", "game_date", "at_bat_number", "pitch_number", "inning", "inning_topbot", "outs_when_up",
    "balls", "strikes", "base", "home_team", "away_team", "bat_team", "fld_team", "home_score", "away_score",
    "bat_score", "fld_score", "description", "plate_x", "plate_z", "sz_top", "sz_bot", "chal_side", "chal_team",
    "w_in", "g_wp", "g_runs", "sigma_in", "challenged", "success", "role", "tokens",
]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(dest: str) -> None:
    dest_dir = pathlib.Path(dest).resolve()
    (dest_dir / "model").mkdir(parents=True, exist_ok=True)
    (dest_dir / "data").mkdir(parents=True, exist_ok=True)

    model = op.Model()
    c = model.c
    sol = op.load_solution()

    # --- shadow prices, checked against the paper's table
    ref = pd.read_csv(OUT / "shadow_prices.csv")
    mine = np.array([[sol["lam"][t, 1, d], sol["lam"][t, 2, d]] for t in range(9 * sol["S"]) for d in range(op.ND)])
    lam_err = float(np.abs(mine - ref[["lam1", "lam2"]].values).max())
    print(f"shadow prices vs outputs/shadow_prices.csv: max abs diff {lam_err:.2e}")
    assert lam_err < 1e-12

    # --- stake table, checked against every called pitch it covers
    g, wp_pre = adv.stake_table()
    stake_err, n_checked = adv.check_stakes(g)

    # --- posterior tables and perception noise
    sig, _, _, _ = model.features(c)
    sigma_mean = {side: float(np.mean(sig[(c.chal_side == side).values])) for side in ("batter", "fielder")}
    tabs = {side: model.tabs[side] for side in ("batter", "fielder")}
    for side, tab in tabs.items():
        assert np.all(np.diff(tab, axis=1) >= -1e-12), f"{side} posterior not monotone in y"
        assert tab.min() >= 0 and tab.max() <= 1

    np.savez_compressed(
        dest_dir / "model" / "research_tables.npz",
        g=g.astype(np.float64), wp_pre=wp_pre.astype(np.float64),
        lam=sol["lam"].astype(np.float64), lamx=sol["lamx"].astype(np.float64),
        S=np.int64(sol["S"]), Sx=np.int64(sol["Sx"]),
        post_batter=tabs["batter"], post_fielder=tabs["fielder"],
        wgrid=op.WGRID, sgrid=op.SGRID, d_home=adv.DH,
    )
    w_prior = {side: c.loc[c.chal_side == side, "w_in"].to_numpy() for side in ("batter", "fielder")}
    np.savez_compressed(dest_dir / "model" / "reference_prior.npz", g=c.g_wp.to_numpy(),
                        **{f"w_{k}": v for k, v in w_prior.items()})

    # --- replay data in game order
    r = c.assign(sigma_in=sig).sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    r = r[REPLAY_COLUMNS].copy()
    r["success"] = r["success"].map({True: 1, False: 0})
    r.to_csv(dest_dir / "data" / "replay_2026.csv.gz", index=False, float_format="%.6g", compression="gzip")

    meta = dict(
        source="ABSGTO (Spending a Scarce Option Under Perceptual Noise)",
        season=2026,
        extra_rule=json.load(open(OUT / "summary_build.json")).get("extra_rule", "reset1_each"),
        S=int(sol["S"]), Sx=int(sol["Sx"]), kmax=int(op.KMAX),
        sigma_mean_in=sigma_mean,
        policy_values=json.load(open(OUT / "policy_values.json"))["optimal"],
        league_gto_wins=json.load(open(OUT / "summary_eval.json"))["league_gto_wins"],
        league_captured_wins=json.load(open(OUT / "summary_eval.json"))["league_captured_wins"],
        checks=dict(lambda_max_abs_diff=lam_err, stake_max_abs_diff=stake_err, stake_pitches_checked=n_checked),
        replay_rows=int(len(r)), replay_games=int(r.game_pk.nunique()),
        sha256=dict(tables=sha256(dest_dir / "model" / "research_tables.npz")),
    )
    (dest_dir / "model" / "research_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
