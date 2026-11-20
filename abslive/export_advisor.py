"""Write web/advisor_data.js, the tables the in-browser advisor reads, from a model version.

    python -m abslive.export_advisor                       # from model/ (the paper's tables)
    MODEL_URI=models:/abs-challenge-policy@champion python -m abslive.export_advisor

The format is the one ABSGTO/src/37_export_advisor.py writes, so web/index.html
needs no changes. For the paper's tables the output is byte-identical to that
script's (checked in tests/test_advisor_export.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import policy
from .registry import ROOT, load_tables

YGRID = np.round(np.arange(-6, 6.01, 0.1), 1)


def advisor_payload(tables: policy.Tables, meta: dict) -> dict:
    post = {side: np.round(policy.posterior(tables, side, YGRID, np.full(len(YGRID), tables.sigma_mean[side])), 4)
            for side in ("batter", "fielder")}
    d_home = tables.d_home
    return dict(
        meta=dict(season=meta["season"], S=tables.S, Sx=tables.Sx, d_home=[int(d_home[0]), int(d_home[-1])],
                  lam_margin=[-4, 4], extra_rule=tables.extra_rule, sigma_in=tables.sigma_mean,
                  y_grid=[float(YGRID[0]), float(YGRID[-1]), 0.1],
                  checks=dict(stake_max_abs_diff=meta["checks"]["stake_max_abs_diff"],
                              stake_pitches_checked=meta["checks"]["stake_pitches_checked"],
                              lambda_max_abs_diff=meta["checks"]["lambda_max_abs_diff"]),
                  **({} if tables.version == "research" else {"model_version": tables.version})),
        g=np.round(tables.g * 1e6).astype(int).ravel().tolist(),
        wp=np.round(tables.wp_pre * 1e4).astype(int).ravel().tolist(),
        lam=np.round(tables.lam[:, 1:3, :] * 1e6).astype(int).tolist(),
        lamx=np.round(tables.lamx[:, 1:] * 1e6).astype(int).tolist(),
        post={k: v.tolist() for k, v in post.items()},
    )


def render(payload: dict) -> str:
    return "window.ADVISOR_DATA=" + json.dumps(payload, separators=(",", ":")) + ";\n"


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "web" / "advisor_data.js"))
    args = ap.parse_args(argv)
    tables, directory = load_tables()
    meta = json.loads((directory / "research_meta.json").read_text())
    out = Path(args.out)
    out.write_text(render(advisor_payload(tables, meta)))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) from {tables.version}")
    return out


if __name__ == "__main__":
    main()
