"""The optimal challenge rule from the ABSGTO paper, as plain numpy.

Everything the stream, the API and the batch replay compute goes through this
file, so they cannot disagree. The rule (paper, Theorem 1):

    challenge  iff  p * g > (1 - p) * lam

    p    probability the call is wrong, given what the player saw (y inches past
         the zone edge) and how noisy his view is (sigma inches)
    g    win probability the call is worth to the challenging team
    lam  shadow price of a challenge: the win probability a team expects to lose
         later by spending one now, given the game state and challenges held

Index conventions follow ABSGTO/src/05_optimal_policy.py exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KMAX = 4
MARGIN_MIN, MARGIN_MAX = -4, 4
SIDES = ("batter", "fielder")


@dataclass(frozen=True)
class Tables:
    g: np.ndarray            # [bot, inning-1 (9 = extras), base, outs, balls, strikes, home lead + 6]
    wp_pre: np.ndarray       # same shape: batting team's win probability before the pitch
    lam: np.ndarray          # [regulation slot, challenges held 0..2, margin + 4]
    lamx: np.ndarray         # [extra-inning slot, challenges held 0..KMAX]
    S: int
    Sx: int
    post: dict               # side -> [sigma grid, y grid] table of P(miss | y, sigma)
    wgrid: np.ndarray
    sgrid: np.ndarray
    d_home: np.ndarray
    sigma_mean: dict         # side -> league-average perception noise (inches)
    extra_rule: str
    version: str = "local"

    @classmethod
    def load(cls, directory: str | Path, version: str = "local") -> "Tables":
        directory = Path(directory)
        z = np.load(directory / "research_tables.npz")
        meta = json.loads((directory / "research_meta.json").read_text())
        post = {"batter": z["post_batter"], "fielder": z["post_fielder"]}
        prior_file = directory / "posterior_override.npz"
        if prior_file.exists():          # written by `abslive.train` when it refits the prior
            o = np.load(prior_file)
            post = {"batter": o["post_batter"], "fielder": o["post_fielder"]}
        return cls(
            g=z["g"], wp_pre=z["wp_pre"], lam=z["lam"], lamx=z["lamx"], S=int(z["S"]), Sx=int(z["Sx"]),
            post=post, wgrid=z["wgrid"], sgrid=z["sgrid"], d_home=z["d_home"],
            sigma_mean=meta["sigma_mean_in"], extra_rule=meta.get("extra_rule", "reset1_each"), version=version,
        )


# ---------------------------------------------------------------- game-state indexes


def margin(chal_side, bat_score, fld_score):
    """The challenging team's lead, clipped to -4..4."""
    side = np.asarray(chal_side)
    lead = np.where(side == "batter", np.asarray(bat_score) - np.asarray(fld_score),
                    np.asarray(fld_score) - np.asarray(bat_score))
    return np.clip(lead, MARGIN_MIN, MARGIN_MAX).astype(int)


def time_index(inning, is_bottom, outs, S):
    t = (np.minimum(np.asarray(inning), 9) - 1) + np.where(is_bottom, 0.5, 0.0) + np.asarray(outs) / 6
    return np.minimum((t * S).astype(int), 9 * S - 1)


def extra_index(is_bottom, outs, Sx):
    f = np.where(is_bottom, 0.5, 0.0) + np.asarray(outs) / 6
    return np.minimum((f * Sx).astype(int), Sx - 1)


def extra_bonus(k, rule="reset1_each"):
    """Challenges available at the start of an extra inning, given k held."""
    k = np.asarray(k)
    if rule == "reset1_each":
        return np.maximum(k, 1)
    if rule == "each_extra":
        return np.minimum(k + 1, KMAX)
    raise ValueError(rule)


# ---------------------------------------------------------------- the three numbers


def posterior(tables: Tables, side, y, sigma):
    """P(call is wrong | perceived miss y, noise sigma), interpolated in sigma as the paper does."""
    side = np.asarray(side)
    y = np.asarray(y, dtype=float)
    sigma = np.broadcast_to(np.maximum(np.asarray(sigma, dtype=float), tables.sgrid[0]), y.shape)
    side = np.broadcast_to(side, y.shape)
    out = np.zeros(y.shape)
    w0, step = tables.wgrid[0], tables.wgrid[1] - tables.wgrid[0]
    for name in SIDES:
        m = side == name
        if not m.any():
            continue
        tab, sg = tables.post[name], tables.sgrid
        s = sigma[m]
        si = np.clip(np.searchsorted(sg, s), 1, len(sg) - 1)
        wi = np.clip(np.round((y[m] - w0) / step).astype(int), 0, len(tables.wgrid) - 1)
        lo, hi = sg[si - 1], sg[si]
        f = np.clip((s - lo) / (hi - lo), 0, 1)
        out[m] = (1 - f) * tab[si - 1, wi] + f * tab[si, wi]
    return out


def shadow_price(tables: Tables, inning, is_bottom, outs, k, lead):
    """lam for k challenges held; regulation uses (slot, k<=2, lead), extras (slot, k<=KMAX)."""
    inning = np.asarray(inning)
    k = np.clip(np.asarray(k), 0, KMAX)
    reg = inning <= 9
    t = time_index(inning, is_bottom, outs, tables.S)
    s = extra_index(is_bottom, outs, tables.Sx)
    d = np.asarray(lead) + 4
    return np.where(reg, tables.lam[t, np.minimum(k, 2), d], tables.lamx[s, k])


def stake(tables: Tables, inning, is_bottom, base, outs, balls, strikes, home_lead):
    """g for the team that would challenge, from the stake table (home lead clipped to the table's range)."""
    lo, hi = int(tables.d_home[0]), int(tables.d_home[-1])
    idx = (np.asarray(is_bottom, int), np.minimum(np.asarray(inning), 10) - 1, np.asarray(base), np.asarray(outs),
           np.asarray(balls), np.asarray(strikes), np.clip(np.asarray(home_lead), lo, hi) - lo)
    return tables.g[idx]


def decide(p, g, lam):
    """Theorem 1, plus the break-even confidence p* = lam / (g + lam)."""
    p, g, lam = np.asarray(p), np.asarray(g), np.asarray(lam)
    challenge = p * g > (1 - p) * lam
    with np.errstate(divide="ignore", invalid="ignore"):
        breakeven = np.where(g + lam > 0, lam / (g + lam), 0.0)
    return challenge, breakeven


# ---------------------------------------------------------------- perception draws


def perceived_miss(w_in, sigma, keys, seed: int = 7):
    """What the player 'saw': the true miss plus his perception noise.

    The noise for a pitch depends only on (seed, game, at-bat, pitch), so the
    streaming job and the batch replay draw exactly the same number no matter
    how the data are split into batches.
    """
    keys = np.asarray(keys, dtype=np.uint64)
    eps = np.array([np.random.default_rng([seed, *map(int, k)]).standard_normal() for k in keys])
    return np.asarray(w_in, float) + eps * np.asarray(sigma, float)
