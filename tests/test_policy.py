"""The decision library against the paper's model and its own invariants."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abslive import policy, replay
from abslive.stake_input import Situation, evaluate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tables():
    return policy.Tables.load(ROOT / "model")


@pytest.fixture(scope="module")
def season():
    return pd.read_csv(ROOT / "data" / "replay_2026.csv.gz")


@pytest.fixture(scope="module")
def few_games(season):
    games = season.game_pk.drop_duplicates().iloc[:40]
    return season[season.game_pk.isin(games)]


def test_export_checks_recorded(tables):
    meta = json.loads((ROOT / "model" / "research_meta.json").read_text())
    assert meta["checks"]["lambda_max_abs_diff"] < 1e-12 and meta["checks"]["stake_max_abs_diff"] == 0
    assert tables.S == 8 and tables.Sx == 7 and tables.extra_rule == "reset1_each"


def test_posterior_is_monotone_and_bounded(tables):
    y = np.linspace(-10, 10, 401)
    for side in policy.SIDES:
        for sigma in (0.5, 1.7, 2.7, 6.0):
            p = policy.posterior(tables, side, y, sigma)
            assert np.all(np.diff(p) >= -1e-12) and p.min() >= 0 and p.max() <= 1
        # sharper eyes are surer either way (value of vision)
        assert policy.posterior(tables, side, 3.0, 1.0) > policy.posterior(tables, side, 3.0, 4.0)


def test_shadow_price_properties(tables):
    """Theorem 2: lam >= 0, and holding fewer challenges makes each one dearer.

    The paper's one documented exception: the final opportunity of regulation, in all
    9 score states, by at most 0.2 points, because extra innings hand a team with no
    challenges a free one.
    """
    assert tables.lam.min() >= -1e-12
    gap = tables.lam[:, 1, :] - tables.lam[:, 2, :]
    assert np.all(gap[:-1] >= -1e-12)
    assert gap[-1].min() >= -0.0021
    assert policy.shadow_price(tables, 1, False, 0, 0, 0) == tables.lam[0, 0, 4]


def test_decision_rule_and_breakeven():
    ch, be = policy.decide(np.array([0.2, 0.6]), np.array([0.05, 0.01]), np.array([0.01, 0.02]))
    assert ch.tolist() == [True, False]
    assert be[0] == pytest.approx(0.01 / 0.06)


def test_stake_lookup_matches_the_season(tables, season):
    s = season[(season.home_score - season.away_score).abs() <= 6].sample(5000, random_state=1)
    g = policy.stake(tables, s.inning.values, (s.inning_topbot == "Bot").values, s.base.values,
                     s.outs_when_up.values, s.balls.values, s.strikes.values,
                     (s.home_score - s.away_score).values)
    assert np.abs(g - s.g_wp.values).max() < 1e-5   # the CSV keeps 6 significant digits


def test_situation_inputs(tables):
    base = dict(inning=9, half="bottom", outs=2, balls=3, strikes=2, on_first=True, on_second=True,
                on_third=True, home_score=3, away_score=3, side="batter", challenges_left=1, perceived_miss_in=1.0)
    out = evaluate(tables, Situation(**base))
    assert out["recommendation"] == "challenge" and out["stake_wp"] > 0.3
    calm = evaluate(tables, Situation(**{**base, "inning": 1, "half": "top", "balls": 0, "strikes": 0,
                                         "on_first": False, "on_second": False, "on_third": False,
                                         "perceived_miss_in": -1.0, "challenges_left": 2}))
    assert calm["recommendation"] == "hold"
    assert evaluate(tables, Situation(**{**base, "challenges_left": 0}))["recommendation"] == "hold"
    with pytest.raises(ValueError):
        Situation(**{**base, "inning": 5, "challenges_left": 3})


def test_draws_do_not_depend_on_batching(few_games):
    keys = few_games[replay.ORDER].to_numpy()
    whole = policy.perceived_miss(few_games.w_in, few_games.sigma_in, keys)
    halves = np.concatenate([policy.perceived_miss(few_games.w_in[:100], few_games.sigma_in[:100], keys[:100]),
                             policy.perceived_miss(few_games.w_in[100:], few_games.sigma_in[100:], keys[100:])])
    assert np.array_equal(whole, halves)


def test_sequential_state_machine_matches_the_vectorised_replay(tables, few_games):
    """replay_frame (what the stream runs) and replay_expected (the paper's loop) agree exactly."""
    scored = replay.replay_frame(tables, few_games)
    srt = few_games.sort_values(replay.ORDER)
    eps = ((policy.perceived_miss(srt.w_in, srt.sigma_in, srt[replay.ORDER].to_numpy()) - srt.w_in.values)
           / srt.sigma_in.values)[:, None]
    total = replay.replay_expected(tables, few_games, R=1, eps=eps)[0]
    assert scored.optimal_gain_wp.sum() == pytest.approx(total, abs=1e-12)
    assert len(scored) == len(few_games)
    # tokens only fall after a lost challenge, and never below zero
    assert scored.tokens_before.min() >= 0
    lost = scored[scored.optimal_challenge & ~scored.ump_wrong]
    assert (lost.tokens_before > 0).all()


def test_actual_value_matches_the_paper(season):
    captured = season.g_wp[season.challenged & (season.success == 1)].sum()
    assert captured == pytest.approx(104.2, abs=0.05)


@pytest.mark.slow
def test_optimal_value_matches_the_paper(tables, season):
    tot = replay.replay_expected(tables, season, R=30, seed=11)
    assert abs(tot.mean() - 140.17) < 4 * tot.std() / np.sqrt(len(tot)) + 0.1
