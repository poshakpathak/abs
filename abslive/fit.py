"""Refit the posterior P(call is wrong | perceived miss, noise) from a window of called pitches.

The paper builds the posterior from the empirical distribution ("prior") of true
misses w, by challenging side, with Gaussian perception noise
(ABSGTO/src/05_optimal_policy.posterior_table). That prior is the part of the
model that moves when umpires change, so it is what retraining refits. The
shadow prices and stakes come from the paper's full-season MDP solution and are
not re-solved here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import policy


def posterior_table(w_prior: np.ndarray, wgrid: np.ndarray, sgrid: np.ndarray) -> np.ndarray:
    """P(w > 0 | y, sigma) on (sigma grid x y grid). Same formula and smoothing as the paper."""
    step = wgrid[1] - wgrid[0]
    edges = np.append(wgrid - step / 2, wgrid[-1] + step / 2)
    hist, _ = np.histogram(np.clip(w_prior, wgrid[0], wgrid[-1]), bins=edges)
    prior = (hist + 1e-3) / (hist + 1e-3).sum()
    pos = wgrid > 0
    tab = np.zeros((len(sgrid), len(wgrid)))
    for i, s in enumerate(sgrid):
        L = norm.pdf((wgrid[:, None] - wgrid[None, :]) / s)
        tab[i] = (L[:, pos] * prior[pos]).sum(1) / np.maximum((L * prior).sum(1), 1e-300)
    return np.clip(np.maximum.accumulate(tab, axis=1), 0.0, 1.0)


def fit_posteriors(tables: policy.Tables, pitches: pd.DataFrame) -> dict[str, np.ndarray]:
    return {side: posterior_table(pitches.loc[pitches.chal_side == side, "w_in"].to_numpy(),
                                  tables.wgrid, tables.sgrid) for side in policy.SIDES}


def calibration(tables: policy.Tables, pitches: pd.DataFrame, seed: int = 3, bins: int = 10) -> dict[str, float]:
    """How well P(call is wrong) matches reality on these pitches, with simulated perception.

    ABS tells us the truth for every pitch, so the posterior can be scored
    directly: Brier score, expected calibration error, and log loss.
    """
    rng = np.random.default_rng(seed)
    y = pitches.w_in.to_numpy() + rng.standard_normal(len(pitches)) * pitches.sigma_in.to_numpy()
    p = policy.posterior(tables, pitches.chal_side.to_numpy(), y, pitches.sigma_in.to_numpy())
    truth = (pitches.w_in.to_numpy() > 0).astype(float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[idx == b].mean() - truth[idx == b].mean()) * (idx == b).mean()
              for b in range(bins) if (idx == b).any())
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(brier=float(np.mean((p - truth) ** 2)), ece=float(ece),
                log_loss=float(-np.mean(truth * np.log(pc) + (1 - truth) * np.log(1 - pc))),
                miss_rate=float(truth.mean()), n=int(len(p)))


def psi(reference: np.ndarray, current: np.ndarray, edges: np.ndarray) -> float:
    """Population stability index of `current` against `reference` on fixed bins."""
    ref, _ = np.histogram(np.clip(reference, edges[0], edges[-1]), edges)
    cur, _ = np.histogram(np.clip(current, edges[0], edges[-1]), edges)
    r = (ref + 0.5) / (ref + 0.5).sum()
    c = (cur + 0.5) / (cur + 0.5).sum()
    return float(np.sum((c - r) * np.log(c / r)))


# Bins for the miss distance near the zone edge, where the decisions are made, and for stakes.
MISS_EDGES = np.array([-40, -12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 40], float)
STAKE_EDGES = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1.0])
