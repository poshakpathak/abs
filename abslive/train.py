"""Build, validate and register a version of the challenge policy in MLflow.

    python -m abslive.train --train-end 2026-06-30            # fit on the first half
    python -m abslive.train --train-end 2026-09-30 --promote  # refit on everything so far

A run:
  1. refits the posterior from called pitches in the training window,
  2. scores it on the pitches after the window (calibration) and replays them
     under the optimal policy (wins captured),
  3. logs parameters, metrics and the model files to MLflow and registers a new
     version of `abs-challenge-policy` with the alias `challenger`,
  4. moves the `champion` alias to it if every gate passes and it is no worse
     than the current champion on the same validation pitches (or with --promote).

Without MLFLOW_TRACKING_URI set, MLflow writes to ./mlruns.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import fit, policy, replay

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "abs-challenge-policy"

GATES = dict(
    max_ece=0.02,               # calibration error on held-out pitches
    max_brier_vs_research=0.002,  # no more than this worse than the paper's own posterior
    min_value_ratio=0.98,       # optimal wins on held-out pitches, relative to the paper's posterior
)


def build_model_dir(tables: policy.Tables, posts: dict, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("research_tables.npz", "research_meta.json"):
        shutil.copy(ROOT / "model" / name, dest / name)
    np.savez_compressed(dest / "posterior_override.npz", post_batter=posts["batter"], post_fielder=posts["fielder"])
    return dest


def save_reference(train: pd.DataFrame, dest: Path) -> None:
    """The training window's distributions, which the drift monitor compares live data against."""
    arrays = {f"w_{side}": train.loc[train.chal_side == side, "w_in"].to_numpy() for side in policy.SIDES}
    np.savez_compressed(dest / "reference.npz", g=train.g_wp.to_numpy(), **arrays)


def evaluate(tables: policy.Tables, valid: pd.DataFrame, R: int) -> dict:
    cal = fit.calibration(tables, valid)
    wins = replay.replay_expected(tables, valid, R=R, seed=11)
    return dict(**{f"valid_{k}": v for k, v in cal.items()},
                valid_optimal_wins=float(wins.mean()), valid_optimal_wins_se=float(wins.std() / np.sqrt(R)))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "replay_2026.csv.gz"))
    ap.add_argument("--train-start", default="2026-01-01")
    ap.add_argument("--train-end", default="2026-06-30")
    ap.add_argument("--valid-end", default="2026-12-31")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--promote", action="store_true", help="make this version champion if the gates pass")
    ap.add_argument("--no-register", action="store_true", help="log the run but do not register a version")
    args = ap.parse_args(argv)

    import mlflow
    from mlflow import MlflowClient

    from .mlflow_model import ChallengePolicy

    season = pd.read_csv(args.data)
    train = season[(season.game_date >= args.train_start) & (season.game_date <= args.train_end)]
    valid = season[(season.game_date > args.train_end) & (season.game_date <= args.valid_end)]
    if len(train) == 0 or len(valid) == 0:
        raise SystemExit("the training or validation window is empty")

    research = policy.Tables.load(ROOT / "model", version="research")
    posts = fit.fit_posteriors(research, train)

    mlflow.set_experiment("abs-challenge-policy")
    with mlflow.start_run(run_name=f"prior-to-{args.train_end}") as run, tempfile.TemporaryDirectory() as tmp:
        model_dir = build_model_dir(research, posts, Path(tmp) / "model_dir")
        save_reference(train, model_dir)
        candidate = policy.Tables.load(model_dir, version="candidate")

        mlflow.log_params(dict(train_start=args.train_start, train_end=args.train_end, valid_end=args.valid_end,
                               train_pitches=len(train), valid_pitches=len(valid), replays=args.replays,
                               noise="normal", sigma_batter=round(research.sigma_mean["batter"], 4),
                               sigma_fielder=round(research.sigma_mean["fielder"], 4)))
        mine = evaluate(candidate, valid, args.replays)
        ref = evaluate(research, valid, args.replays)
        for side in policy.SIDES:
            mine[f"posterior_max_abs_diff_vs_research_{side}"] = float(np.abs(candidate.post[side] - research.post[side]).max())
        mine["train_miss_rate"] = float((train.w_in > 0).mean())
        mine["psi_miss_train_vs_valid"] = fit.psi(train.w_in.values, valid.w_in.values, fit.MISS_EDGES)
        mine["research_valid_brier"] = ref["valid_brier"]
        mine["research_valid_optimal_wins"] = ref["valid_optimal_wins"]
        mlflow.log_metrics(mine)

        gates = {
            "ece": mine["valid_ece"] <= GATES["max_ece"],
            "brier": mine["valid_brier"] - ref["valid_brier"] <= GATES["max_brier_vs_research"],
            "value": mine["valid_optimal_wins"] >= GATES["min_value_ratio"] * ref["valid_optimal_wins"],
        }
        mlflow.log_dict(dict(gates=GATES, passed=gates), "gates.json")
        mlflow.set_tag("gates_passed", str(all(gates.values())).lower())

        info = mlflow.pyfunc.log_model(
            name="model", python_model=ChallengePolicy(), artifacts={"model_dir": str(model_dir)},
            code_paths=[str(ROOT / "abslive")],
            input_example=pd.DataFrame([dict(inning=9, half="bottom", outs=2, balls=3, strikes=2, on_first=True,
                                              on_second=False, on_third=False, home_score=3, away_score=3,
                                              side="batter", challenges_left=1, perceived_miss_in=1.0)]),
            pip_requirements=["numpy", "pandas", "scipy", "pydantic>=2"],
        )
        result = dict(run_id=run.info.run_id, metrics=mine, gates=gates, version=None, champion=False)
        if args.no_register:
            print(json.dumps(result, indent=2, default=str))
            return result

        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
        client = MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
        for k in ("train_end", "valid_end"):
            client.set_model_version_tag(MODEL_NAME, mv.version, k, getattr(args, k))
        client.set_model_version_tag(MODEL_NAME, mv.version, "gates_passed", str(all(gates.values())).lower())
        result["version"] = int(mv.version)

        try:
            champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        except Exception:
            champ = None
        promote = all(gates.values()) and (args.promote or champ is None)
        if all(gates.values()) and champ is not None and not args.promote:
            champ_tables = load_registered_tables(f"models:/{MODEL_NAME}/{champ.version}", str(champ.version))
            champ_metrics = evaluate(champ_tables, valid, args.replays)
            promote = mine["valid_brier"] <= champ_metrics["valid_brier"] + 1e-6
            mlflow.log_metric("champion_valid_brier", champ_metrics["valid_brier"])
        if promote:
            client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
            result["champion"] = True
        print(json.dumps(result, indent=2, default=str))
        return result


def load_registered_tables(model_uri: str, version: str) -> policy.Tables:
    """Download a registered model's files and load its tables."""
    import mlflow

    local = Path(mlflow.artifacts.download_artifacts(model_uri))
    found = list(local.rglob("research_tables.npz"))
    if not found:
        raise FileNotFoundError(f"no model files under {local}")
    return policy.Tables.load(found[0].parent, version=version)


if __name__ == "__main__":
    main()
