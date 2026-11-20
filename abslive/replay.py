"""Replay a season's called pitches under the optimal policy, one team-game at a time.

`TeamGameState` is the per-team-game state the streaming job keeps (challenges
held, last extra inning that granted one). `replay_frame` runs the same state
machine over a whole DataFrame in plain pandas, and is what the stream's output
is tested against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import policy

ORDER = ["game_pk", "at_bat_number", "pitch_number"]


@dataclass
class TeamGameState:
    tokens: int = 2
    last_bonus_inning: int = 0
    captured_wp: float = 0.0     # optimal policy: win probability won back
    challenges: int = 0
    overturns: int = 0


def score_pitches(tables: policy.Tables, df: pd.DataFrame, state: TeamGameState, seed: int = 7) -> pd.DataFrame:
    """Score one team-game's pitches in order, updating `state` in place. Returns one row per pitch."""
    df = df.sort_values(ORDER)
    rows = []
    bottom = (df.inning_topbot.values == "Bot")
    lead = policy.margin(df.chal_side.values, df.bat_score.values, df.fld_score.values)
    y = policy.perceived_miss(df.w_in.values, df.sigma_in.values,
                              df[ORDER].to_numpy(), seed=seed)
    p_all = policy.posterior(tables, df.chal_side.values, y, df.sigma_in.values)
    # The shadow price at the challenges the team actually held: what a failed real challenge cost.
    lam_actual = policy.shadow_price(tables, df.inning.values, bottom, df.outs_when_up.values,
                                     df.tokens.values, lead)
    for i, r in enumerate(df.itertuples(index=False)):
        inn = int(r.inning)
        if inn >= 10 and inn > state.last_bonus_inning:
            state.tokens = int(policy.extra_bonus(state.tokens, tables.extra_rule))
            state.last_bonus_inning = inn
        k = state.tokens
        lam = float(policy.shadow_price(tables, inn, bottom[i], r.outs_when_up, k, lead[i]))
        p = float(p_all[i]); g = float(r.g_wp)
        challenge, breakeven = policy.decide(p, g, lam)
        challenge = bool(challenge) and k > 0
        right = r.w_in > 0
        if challenge:
            state.challenges += 1
            if right:
                state.overturns += 1
                state.captured_wp += g
            else:
                state.tokens = k - 1
        actual_success = bool(r.success == 1)
        rows.append(dict(
            game_pk=int(r.game_pk), at_bat_number=int(r.at_bat_number), pitch_number=int(r.pitch_number),
            game_date=str(r.game_date), chal_team=r.chal_team, chal_side=r.chal_side, inning=inn,
            tokens_before=k, tokens_actual=int(r.tokens),
            w_in=float(r.w_in), y_in=float(y[i]), sigma_in=float(r.sigma_in), p_miss=p, g_wp=g, lam=lam,
            breakeven_p=float(breakeven),
            optimal_challenge=challenge, ump_wrong=bool(right),
            optimal_gain_wp=g if (challenge and right) else 0.0,
            actual_challenge=bool(r.challenged), actual_success=actual_success,
            actual_gain_wp=g if (r.challenged and actual_success) else 0.0,
            # Challenge Value Added (paper, 06_evaluate): g for an overturn, minus the shadow price of a lost one.
            actual_cva_wp=(g if actual_success else -float(lam_actual[i])) if r.challenged else 0.0,
            model_version=tables.version,
        ))
    return pd.DataFrame(rows)


def replay_frame(tables: policy.Tables, df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Score every pitch of every team-game. The streaming job must match this row for row."""
    out = []
    for (_, _), part in df.groupby(["game_pk", "chal_team"], sort=False):
        out.append(score_pitches(tables, part, TeamGameState(), seed=seed))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def replay_expected(tables: policy.Tables, df: pd.DataFrame, R: int = 200, seed: int = 7,
                    eps: np.ndarray | None = None) -> np.ndarray:
    """League wins captured by the optimal policy, for R independent perception draws (vectorised).

    Mirrors ABSGTO/src/06_evaluate.replay(policy="optimal"). Returns one total per draw.
    `eps`, if given, is the noise to use instead of fresh draws: shape (rows, R), in the
    order of `df` sorted by game, at-bat and pitch.
    """
    rng = np.random.default_rng(seed)
    c = df.sort_values(ORDER).copy()
    c["row"] = np.arange(len(c))
    c["tg"] = c.game_pk.astype(str) + "_" + c.chal_team
    c["j"] = c.groupby("tg").cumcount()
    c["gi"] = pd.factorize(c.tg)[0]
    G = c.gi.max() + 1
    tok = np.full((G, R), 2)
    last_bonus = np.zeros(G, int)
    gain = np.zeros((G, R))
    bottom = (c.inning_topbot.values == "Bot")
    c["lead"] = policy.margin(c.chal_side.values, c.bat_score.values, c.fld_score.values)
    c["bottom"] = bottom
    for _, s in c.groupby("j", sort=True):
        gi = s.gi.values; n = len(s); inn = s.inning.values
        newx = (inn >= 10) & (inn > last_bonus[gi])
        if newx.any():
            tok[gi[newx]] = policy.extra_bonus(tok[gi[newx]], tables.extra_rule)
            last_bonus[gi[newx]] = inn[newx]
        k = tok[gi]
        w = s.w_in.values[:, None]; right = w > 0
        sg = s.sigma_in.values[:, None]
        noise = rng.standard_normal((n, R)) if eps is None else eps[s.row.values]
        y = w + noise * sg
        p = policy.posterior(tables, np.repeat(s.chal_side.values[:, None], R, 1), y, np.broadcast_to(sg, y.shape))
        L = policy.shadow_price(tables, inn[:, None], s.bottom.values[:, None], s.outs_when_up.values[:, None],
                                k, s.lead.values[:, None])
        g = s.g_wp.values[:, None]
        ch = (k > 0) & (p * g > (1 - p) * L)
        win = ch & right
        gain[gi] += np.where(win, g, 0)
        tok[gi] = np.where(ch & ~right, k - 1, k)
    return gain.sum(0)
