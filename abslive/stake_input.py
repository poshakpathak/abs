"""Turn a game situation, as a person would describe it, into the paper's state variables."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from . import policy


class Situation(BaseModel):
    """A called pitch the challenging side might dispute."""

    inning: int = Field(ge=1, le=25)
    half: Literal["top", "bottom"]
    outs: int = Field(ge=0, le=2)
    balls: int = Field(ge=0, le=3, description="Count before the pitch")
    strikes: int = Field(ge=0, le=2, description="Count before the pitch")
    on_first: bool = False
    on_second: bool = False
    on_third: bool = False
    home_score: int = Field(ge=0, le=50)
    away_score: int = Field(ge=0, le=50)
    side: Literal["batter", "fielder"] = Field(
        description="batter disputes a called strike; fielder (catcher or pitcher) disputes a called ball")
    challenges_left: int = Field(ge=0, le=policy.KMAX)
    perceived_miss_in: float = Field(
        ge=-40, le=40,
        description="How far past the zone edge the player saw the ball, in inches; positive means he thinks the call is wrong")
    sigma_in: float | None = Field(
        default=None, gt=0, le=12,
        description="The player's perception noise in inches; league average for the side if omitted")

    @model_validator(mode="after")
    def _regulation_tokens(self):
        if self.inning <= 9 and self.challenges_left > 2:
            raise ValueError("a team holds at most 2 challenges in regulation")
        return self

    # --- derived state

    @property
    def is_bottom(self) -> bool:
        return self.half == "bottom"

    @property
    def base(self) -> int:
        return int(self.on_first) + 2 * int(self.on_second) + 4 * int(self.on_third)

    @property
    def lead(self) -> int:
        """The challenging team's lead."""
        bat, fld = (self.home_score, self.away_score) if self.is_bottom else (self.away_score, self.home_score)
        return int(policy.margin(self.side, bat, fld))


def evaluate(tables: policy.Tables, s: Situation) -> dict:
    sigma = s.sigma_in if s.sigma_in is not None else tables.sigma_mean[s.side]
    p = float(policy.posterior(tables, s.side, s.perceived_miss_in, sigma))
    g = float(policy.stake(tables, s.inning, s.is_bottom, s.base, s.outs, s.balls, s.strikes,
                           s.home_score - s.away_score))
    k = s.challenges_left
    lam = float(policy.shadow_price(tables, s.inning, s.is_bottom, s.outs, k, s.lead))
    challenge, breakeven = policy.decide(p, g, lam)
    challenge = bool(challenge) and k > 0
    return dict(
        recommendation="challenge" if challenge else "hold",
        p_call_wrong=round(p, 4), breakeven_confidence=round(float(breakeven), 4),
        stake_wp=round(g, 5), shadow_price_wp=round(lam, 5),
        expected_gain_wp=round(p * g - (1 - p) * lam, 5) if k > 0 else 0.0,
        sigma_in=round(float(sigma), 3), model_version=tables.version,
    )
