"""HTTP API for the challenge policy.

    uvicorn service.app:app --port 8000

    POST /advise         one situation -> challenge or hold, with the numbers behind it
    POST /advise/batch   up to 500 situations
    GET  /model          which model version is being served, and where it came from
    GET  /health
    GET  /metrics        Prometheus metrics
    POST /reload         pick up a newly promoted champion (needs ADMIN_TOKEN)

The model is the MLflow registry's champion when MODEL_URI is set
(e.g. models:/abs-challenge-policy@champion), else the paper's tables in model/.
"""
from __future__ import annotations

import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from pydantic import BaseModel, Field

from abslive import __version__
from abslive.registry import load_tables
from abslive.stake_input import Situation, evaluate

REQUESTS = Counter("abs_api_requests_total", "Requests by route and status", ["route", "status"])
LATENCY = Histogram("abs_api_latency_seconds", "Request latency", ["route"],
                    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0))
DECISIONS = Counter("abs_api_decisions_total", "Recommendations served", ["side", "recommendation"])
P_MISS = Histogram("abs_api_p_call_wrong", "P(call is wrong) of served situations",
                   buckets=tuple(i / 10 for i in range(1, 11)))
MODEL_INFO = Info("abs_api_model", "Model being served")
MODEL_LOADED = Gauge("abs_api_model_loaded_timestamp_seconds", "When the current model was loaded")


class State:
    tables = None
    directory = None


def _load() -> None:
    State.tables, State.directory = load_tables()
    MODEL_INFO.info({"version": State.tables.version, "source": os.environ.get("MODEL_URI") or "model/"})
    MODEL_LOADED.set(time.time())


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load()
    yield


app = FastAPI(title="ABS challenge advisor", version=__version__, lifespan=lifespan,
              description="Challenge or hold, for any MLB ABS situation, from the optimal policy solved on "
                          "every 2026 regular-season pitch.")


@app.middleware("http")
async def measure(request: Request, call_next):
    route = request.scope.get("route")
    start = time.perf_counter()
    response = await call_next(request)
    path = getattr(request.scope.get("route"), "path", None) or (route.path if route else "unmatched")
    if path != "/metrics":
        LATENCY.labels(path).observe(time.perf_counter() - start)
        REQUESTS.labels(path, str(response.status_code)).inc()
    return response


class Advice(BaseModel):
    recommendation: str = Field(description="challenge or hold")
    p_call_wrong: float = Field(description="Probability the call is wrong, given what the player saw")
    breakeven_confidence: float = Field(description="Challenge when p_call_wrong is above this")
    stake_wp: float = Field(description="Win probability the call is worth")
    shadow_price_wp: float = Field(description="Win probability a lost challenge costs later")
    expected_gain_wp: float
    sigma_in: float = Field(description="Perception noise used, inches")
    model_version: str


def _serve(s: Situation) -> dict:
    out = evaluate(State.tables, s)
    DECISIONS.labels(s.side, out["recommendation"]).inc()
    P_MISS.observe(out["p_call_wrong"])
    return out


@app.post("/advise", response_model=Advice)
def advise(situation: Situation) -> dict:
    return _serve(situation)


class Batch(BaseModel):
    situations: list[Situation] = Field(min_length=1, max_length=500)


@app.post("/advise/batch", response_model=list[Advice])
def advise_batch(batch: Batch) -> list[dict]:
    return [_serve(s) for s in batch.situations]


@app.get("/model")
def model() -> dict:
    t = State.tables
    return dict(version=t.version, source=os.environ.get("MODEL_URI") or "model/", slots=t.S, extra_slots=t.Sx,
                extra_rule=t.extra_rule, sigma_mean_in=t.sigma_mean)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": State.tables.version}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _admin(authorization: str = Header(default="")) -> None:
    token = os.environ.get("ADMIN_TOKEN", "")
    if not token or not secrets.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="needs Authorization: Bearer $ADMIN_TOKEN")


@app.post("/reload", dependencies=[Depends(_admin)])
def reload() -> dict:
    before = State.tables.version
    _load()
    return {"before": before, "after": State.tables.version}
