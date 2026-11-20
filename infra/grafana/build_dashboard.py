"""Write infra/grafana/dashboards/abs-live.json. Edit the panels here, not in the JSON."""
import json
from pathlib import Path

PG = {"type": "grafana-postgresql-datasource", "uid": "abs-pg"}
PROM = {"type": "prometheus", "uid": "abs-prom"}
_id = iter(range(1, 100))


def sql(query, fmt="table"):
    return {"datasource": PG, "refId": "A", "rawQuery": True, "editorMode": "code", "format": fmt, "rawSql": query}


def prom(expr, legend="", ref="A", instant=False):
    return {"datasource": PROM, "refId": ref, "expr": expr, "legendFormat": legend, "instant": instant,
            "range": not instant}


def panel(kind, title, x, y, w, h, targets, ds, **extra):
    p = {"id": next(_id), "type": kind, "title": title, "datasource": ds,
         "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets}
    p.update(extra)
    return p


def stat(title, x, y, w, query, unit="none", decimals=None, description=""):
    defaults = {"unit": unit, "color": {"mode": "fixed", "fixedColor": "#EFC75E"}}
    if decimals is not None:
        defaults["decimals"] = decimals
    return panel("stat", title, x, y, w, 4, [sql(query)], PG, description=description,
                 fieldConfig={"defaults": defaults, "overrides": []},
                 options={"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                          "colorMode": "value", "graphMode": "none", "textMode": "value"})


def row(title, y):
    return {"id": next(_id), "type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}}


panels = [
    row("Season, as it streams in", 0),
    stat("Pitches scored", 0, 1, 4, "SELECT count(*) FROM decisions"),
    stat("Games", 4, 1, 4, "SELECT count(DISTINCT game_pk) FROM decisions"),
    stat("Wins captured by players", 8, 1, 4, "SELECT coalesce(sum(actual_gain_wp), 0) FROM decisions", decimals=2,
         description="Win probability won back by successful challenges, summed."),
    stat("Wins captured, optimal policy", 12, 1, 4, "SELECT coalesce(sum(optimal_gain_wp), 0) FROM decisions",
         decimals=2, description="The same pitches, decided by the paper's optimal policy with simulated perception."),
    stat("Wins left on the table", 16, 1, 4,
         "SELECT coalesce(sum(optimal_gain_wp) - sum(actual_gain_wp), 0) FROM decisions", decimals=2),
    stat("Challenges: players / optimal", 20, 1, 4,
         "SELECT count(*) FILTER (WHERE actual_challenge) || ' / ' || count(*) FILTER (WHERE optimal_challenge) "
         "FROM decisions", unit="string"),
    panel("timeseries", "Wins captured over the season", 0, 5, 14, 9, [sql(
        "SELECT game_date::timestamptz AS \"time\",\n"
        "  sum(sum(actual_gain_wp)) OVER (ORDER BY game_date) AS \"Players\",\n"
        "  sum(sum(optimal_gain_wp)) OVER (ORDER BY game_date) AS \"Optimal policy\"\n"
        "FROM decisions WHERE $__timeFilter(game_date::timestamptz)\nGROUP BY game_date ORDER BY 1", "time_series")],
        PG, fieldConfig={"defaults": {"custom": {"lineWidth": 2, "fillOpacity": 8}, "decimals": 2}, "overrides": []}),
    panel("barchart", "Challenges by inning", 14, 5, 10, 9, [sql(
        "SELECT CASE WHEN inning >= 10 THEN 'X' ELSE inning::text END AS inning,\n"
        "  count(*) FILTER (WHERE actual_challenge) AS \"Players\",\n"
        "  count(*) FILTER (WHERE optimal_challenge) AS \"Optimal policy\"\n"
        "FROM decisions GROUP BY 1 ORDER BY min(least(inning, 10))")], PG,
        description="Players hoard: the optimal policy spends far more challenges late.",
        options={"xField": "inning", "orientation": "vertical", "legend": {"showLegend": True, "displayMode": "list"}}),
    panel("table", "Teams", 0, 14, 12, 10, [sql(
        "SELECT chal_team AS \"Team\",\n"
        "  count(*) FILTER (WHERE actual_challenge) AS \"Challenges\",\n"
        "  count(*) FILTER (WHERE optimal_challenge) AS \"Optimal\",\n"
        "  round(sum(actual_gain_wp)::numeric, 2) AS \"Wins captured\",\n"
        "  round(sum(optimal_gain_wp)::numeric, 2) AS \"Optimal wins\",\n"
        "  round((sum(optimal_gain_wp) - sum(actual_gain_wp))::numeric, 2) AS \"Gap\",\n"
        "  round(sum(actual_cva_wp)::numeric, 2) AS \"Net CVA\"\n"
        "FROM decisions GROUP BY 1 ORDER BY \"Gap\" DESC")], PG,
        description="Net CVA: wins from overturns minus the shadow price of each lost challenge (paper, 06_evaluate)."),
    panel("table", "Latest calls where the model and the players disagreed", 12, 14, 12, 10, [sql(
        "SELECT game_date AS \"Date\", chal_team AS \"Team\", inning AS \"Inn\", chal_side AS \"Side\",\n"
        "  round(p_miss::numeric, 3) AS \"P(wrong)\", round(breakeven_p::numeric, 3) AS \"Break-even\",\n"
        "  round((g_wp * 100)::numeric, 2) AS \"Stake (pts)\",\n"
        "  CASE WHEN optimal_challenge THEN 'challenge' ELSE 'hold' END AS \"Model\",\n"
        "  CASE WHEN actual_challenge THEN 'challenged' ELSE 'held' END AS \"Players\",\n"
        "  CASE WHEN ump_wrong THEN 'yes' ELSE 'no' END AS \"Ump wrong\"\n"
        "FROM decisions WHERE optimal_challenge <> actual_challenge\n"
        "ORDER BY scored_at DESC, game_pk DESC, at_bat_number DESC, pitch_number DESC LIMIT 25")], PG),
    row("Model and pipeline health (last 15 minutes)", 24),
    panel("timeseries", "Drift: PSI against the training window", 0, 25, 9, 8, [
        prom('abs_drift_psi{feature="miss"}', "miss distance, {{side}}", "A"),
        prom('abs_drift_psi{feature="stake"}', "stake", "B")], PROM, timeFrom="15m",
        description="Population stability index. Above 0.2 the monitor raises an alert.",
        fieldConfig={"defaults": {"thresholds": {"mode": "absolute", "steps": [
            {"color": "green", "value": None}, {"color": "red", "value": 0.2}]},
            "custom": {"thresholdsStyle": {"mode": "line"}}}, "overrides": []}),
    panel("stat", "Drift alert", 9, 25, 3, 4, [prom("abs_drift_alert", instant=True)], PROM, timeFrom="15m",
          fieldConfig={"defaults": {"mappings": [{"type": "value", "options": {
              "0": {"text": "OK", "color": "green"}, "1": {"text": "ALERT", "color": "red"}}}],
              "color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [
                  {"color": "green", "value": None}, {"color": "red", "value": 1}]}}, "overrides": []},
          options={"colorMode": "background", "graphMode": "none",
                   "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}),
    panel("stat", "Calibration error (ECE)", 9, 29, 3, 4, [prom('abs_calibration{metric="ece"}', instant=True)],
          PROM, timeFrom="15m", fieldConfig={"defaults": {"decimals": 4}, "overrides": []},
          description="How far P(call is wrong) is from what ABS ruled, on the live window."),
    panel("timeseries", "Stream throughput", 12, 25, 6, 8, [
        prom("rate(abs_stream_pitches_total[1m])", "pitches / s")], PROM, timeFrom="15m"),
    panel("timeseries", "API latency, p95", 18, 25, 6, 8, [
        prom("histogram_quantile(0.95, sum(rate(abs_api_latency_seconds_bucket[1m])) by (le, route))", "{{route}}")],
        PROM, timeFrom="15m", fieldConfig={"defaults": {"unit": "s"}, "overrides": []}),
    panel("table", "Model versions in service", 0, 33, 12, 4, [
        prom("abs_api_model_info", "", "A", instant=True)], PROM, timeFrom="15m",
        transformations=[{"id": "labelsToFields", "options": {}},
                         {"id": "organize", "options": {"excludeByName": {"Time": True, "Value": True, "__name__": True}}}]),
]

dashboard = {
    "uid": "abs-live", "title": "ABS challenges, live", "tags": ["abs", "mlops"], "schemaVersion": 39,
    "version": 1, "editable": True, "refresh": "5s", "timezone": "utc",
    "time": {"from": "2026-03-20T00:00:00Z", "to": "2026-10-05T00:00:00Z"},
    "panels": panels,
}
out = Path(__file__).parent / "dashboards" / "abs-live.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out}, {len(panels)} panels")
