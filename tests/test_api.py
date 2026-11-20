import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("prometheus_client")

from fastapi.testclient import TestClient  # noqa: E402

from service.app import app  # noqa: E402

SITUATION = dict(inning=9, half="bottom", outs=2, balls=3, strikes=2, on_first=True, on_second=True,
                 on_third=True, home_score=3, away_score=3, side="batter", challenges_left=1, perceived_miss_in=1.0)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_advise(client):
    r = client.post("/advise", json=SITUATION)
    assert r.status_code == 200
    body = r.json()
    assert body["recommendation"] == "challenge" and body["model_version"] == "research"
    assert 0 < body["breakeven_confidence"] < body["p_call_wrong"] <= 1


def test_matches_the_screenshot_example(client):
    """Bottom 7th, 2-2, one out, runner on second, down 1, batter, 2 left (the advisor's own example):
    the call is worth 7.89 points, losing a challenge costs 0.62, break-even 7%."""
    r = client.post("/advise", json=dict(inning=7, half="bottom", outs=1, balls=2, strikes=2, on_second=True,
                                         home_score=2, away_score=3, side="batter", challenges_left=2,
                                         perceived_miss_in=-1.5)).json()
    assert r["stake_wp"] == pytest.approx(0.0789, abs=0.00005)
    assert r["shadow_price_wp"] == pytest.approx(0.0062, abs=0.00005)
    assert round(r["breakeven_confidence"] * 100) == 7


def test_validation(client):
    assert client.post("/advise", json={**SITUATION, "inning": 3, "challenges_left": 3}).status_code == 422
    assert client.post("/advise", json={**SITUATION, "side": "umpire"}).status_code == 422
    assert client.post("/advise/batch", json={"situations": []}).status_code == 422


def test_batch_and_metrics(client):
    r = client.post("/advise/batch", json={"situations": [SITUATION, {**SITUATION, "challenges_left": 0}]})
    assert [a["recommendation"] for a in r.json()] == ["challenge", "hold"]
    text = client.get("/metrics").text
    assert "abs_api_requests_total" in text and 'abs_api_decisions_total{recommendation="hold",side="batter"}' in text
    assert 'route="/advise/batch"' in text


def test_model_health_and_reload_needs_token(client, monkeypatch):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model").json()["slots"] == 8
    assert client.post("/reload").status_code == 401
    monkeypatch.setenv("ADMIN_TOKEN", "t0ken")
    assert client.post("/reload", headers={"Authorization": "Bearer t0ken"}).json()["after"] == "research"
