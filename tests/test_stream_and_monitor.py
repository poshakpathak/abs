"""Pieces of the stream and the monitor that need no services."""
import numpy as np
import pandas as pd
import pytest

from abslive import fit, monitor, policy
from abslive.replay import TeamGameState, replay_frame, score_pitches
from stream import producer
from stream.spark_job import COLUMNS, OUT_FIELDS, PITCH_FIELDS, plain_rows

from .test_policy import ROOT


@pytest.fixture(scope="module")
def tables():
    return policy.Tables.load(ROOT / "model")


@pytest.fixture(scope="module")
def games():
    s = pd.read_csv(ROOT / "data" / "replay_2026.csv.gz")
    return s[s.game_pk.isin(s.game_pk.drop_duplicates().iloc[:12])]


def test_schemas_cover_the_data_and_the_output(tables, games):
    assert {n for n, _ in PITCH_FIELDS} == set(games.columns)
    out = replay_frame(tables, games.head(50))
    assert list(out.columns) == COLUMNS == [n for n, _ in OUT_FIELDS]
    first = plain_rows(out)[0]
    assert all(type(v) in (int, float, str, bool) for v in first)


def test_state_carries_across_micro_batches(tables, games):
    """Scoring a team-game in two pieces with saved state equals scoring it whole."""
    one = games[(games.game_pk == games.game_pk.iloc[0]) & (games.chal_team == games.chal_team.iloc[0])]
    whole = score_pitches(tables, one, TeamGameState())
    st = TeamGameState()
    a = score_pitches(tables, one.iloc[: len(one) // 2], st)
    b = score_pitches(tables, one.iloc[len(one) // 2:], st)
    pd.testing.assert_frame_equal(whole, pd.concat([a, b], ignore_index=True))


def test_interleave_keeps_each_game_in_order():
    s = pd.read_csv(ROOT / "data" / "replay_2026.csv.gz", usecols=["game_date", "game_pk", "at_bat_number", "pitch_number"])
    busiest = s.groupby("game_date").game_pk.nunique().idxmax()
    day = s[s.game_date == busiest]
    mixed = producer.interleave(day)
    assert len(mixed) == len(day) and mixed.game_pk.iloc[:day.game_pk.nunique()].nunique() > 1
    for _, g in mixed.groupby("game_pk", sort=False):
        assert list(zip(g.at_bat_number, g.pitch_number)) == sorted(zip(g.at_bat_number, g.pitch_number))


def test_monitor_is_quiet_on_real_data_and_fires_on_a_shift(tables, games):
    ref = np.load(ROOT / "model" / "reference_prior.npz")
    reference = {k: ref[k] for k in ref.files}
    scored = replay_frame(tables, games)
    quiet = monitor.check(scored, reference)
    assert quiet["alert"] == 0 and quiet["psi_miss_batter"] < 0.05 and quiet["ece"] < 0.03
    shifted = scored.assign(w_in=scored.w_in + 3.0, ump_wrong=(scored.w_in + 3.0) > 0)
    loud = monitor.check(shifted, reference)
    assert loud["alert"] == 1 and loud["psi_miss_fielder"] > 0.2


def test_psi_basics():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert fit.psi(a, rng.normal(size=20000), fit.MISS_EDGES) < 0.01
    assert fit.psi(a, rng.normal(2, 1, size=20000), fit.MISS_EDGES) > 0.5
