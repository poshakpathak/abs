"""Score every pitch as it arrives: Spark Structured Streaming from Redpanda to Postgres.

    python -m stream.spark_job

For each team in each game the job keeps the optimal policy's state between
micro-batches (challenges held, extra-inning bonuses) with
applyInPandasWithState, and scores each called pitch with the same library the
API and the batch replay use. Rows go to the `decisions` table; inserting is
idempotent, so a restart from the checkpoint never double-counts.
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd

from abslive.replay import TeamGameState, score_pitches

log = logging.getLogger("stream")

PITCH_FIELDS = [
    ("game_pk", "long"), ("game_date", "string"), ("at_bat_number", "int"), ("pitch_number", "int"),
    ("inning", "int"), ("inning_topbot", "string"), ("outs_when_up", "int"), ("balls", "int"), ("strikes", "int"),
    ("base", "int"), ("home_team", "string"), ("away_team", "string"), ("bat_team", "string"),
    ("fld_team", "string"), ("home_score", "int"), ("away_score", "int"), ("bat_score", "int"),
    ("fld_score", "int"), ("description", "string"), ("plate_x", "double"), ("plate_z", "double"),
    ("sz_top", "double"), ("sz_bot", "double"), ("chal_side", "string"), ("chal_team", "string"),
    ("w_in", "double"), ("g_wp", "double"), ("g_runs", "double"), ("sigma_in", "double"),
    ("challenged", "boolean"), ("success", "double"), ("role", "string"), ("tokens", "int"),
]
OUT_FIELDS = [
    ("game_pk", "long"), ("at_bat_number", "int"), ("pitch_number", "int"), ("game_date", "string"),
    ("chal_team", "string"), ("chal_side", "string"), ("inning", "int"), ("tokens_before", "int"),
    ("tokens_actual", "int"), ("w_in", "double"), ("y_in", "double"), ("sigma_in", "double"),
    ("p_miss", "double"), ("g_wp", "double"), ("lam", "double"), ("breakeven_p", "double"),
    ("optimal_challenge", "boolean"), ("ump_wrong", "boolean"), ("optimal_gain_wp", "double"),
    ("actual_challenge", "boolean"), ("actual_success", "boolean"), ("actual_gain_wp", "double"),
    ("actual_cva_wp", "double"), ("model_version", "string"),
]
STATE_FIELDS = [("tokens", "int"), ("last_bonus_inning", "int"), ("captured_wp", "double"),
                ("challenges", "int"), ("overturns", "int")]
COLUMNS = [name for name, _ in OUT_FIELDS]

_TABLES = None


def tables():
    """Load the model once per Python worker. MODEL_DIR is set by the driver after it resolves the champion."""
    global _TABLES
    if _TABLES is None:
        from abslive.policy import Tables

        _TABLES = Tables.load(os.environ["MODEL_DIR"], version=os.environ.get("MODEL_VERSION", "research"))
    return _TABLES


def score_group(key, batches, state):
    """applyInPandasWithState: one team in one game, all its new pitches in this micro-batch."""
    if state.exists:
        tokens, last_bonus, captured, challenges, overturns = state.get
        st = TeamGameState(tokens, last_bonus, captured, challenges, overturns)
    else:
        st = TeamGameState()
    pdf = pd.concat(list(batches), ignore_index=True)
    out = score_pitches(tables(), pdf, st, seed=int(os.environ.get("PERCEPTION_SEED", "7")))
    state.update((st.tokens, st.last_bonus_inning, st.captured_wp, st.challenges, st.overturns))
    yield out[COLUMNS]


def ddl(fields) -> str:
    return ", ".join(f"{name} {kind}" for name, kind in fields)


def plain_rows(df: pd.DataFrame) -> list[tuple]:
    """Rows as plain Python values (psycopg cannot bind numpy scalars), NaN as NULL."""
    out = []
    for row in df[COLUMNS].itertuples(index=False, name=None):
        out.append(tuple(None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, "item") else v)
                         for v in row))
    return out


def make_writer(dsn: str, metrics: dict):
    import psycopg

    placeholders = ", ".join(["%s"] * len(COLUMNS))
    sql = (f"INSERT INTO decisions ({', '.join(COLUMNS)}) VALUES ({placeholders}) "
           "ON CONFLICT (game_pk, at_bat_number, pitch_number) DO NOTHING")

    def write(batch_df, batch_id: int) -> None:
        started = time.time()
        rows = batch_df.toPandas()
        if len(rows):
            with psycopg.connect(dsn) as conn, conn.cursor() as cur:
                cur.executemany(sql, plain_rows(rows))
        seconds = time.time() - started
        metrics["rows"].inc(len(rows))
        metrics["batches"].inc()
        metrics["batch_seconds"].observe(seconds)
        metrics["last_batch_rows"].set(len(rows))
        log.info("batch %d: %d pitches written in %.2fs", batch_id, len(rows), seconds)

    return write


def main() -> None:  # pragma: no cover - run by the integration tests in CI
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.streaming.state import GroupStateTimeout

    from abslive.registry import resolve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    model_dir, version = resolve()
    os.environ["MODEL_DIR"], os.environ["MODEL_VERSION"] = str(model_dir), version
    log.info("scoring with %s", version)

    metrics = dict(
        rows=Counter("abs_stream_pitches_total", "Pitches scored and written"),
        batches=Counter("abs_stream_batches_total", "Micro-batches written"),
        batch_seconds=Histogram("abs_stream_write_seconds", "Time to write a micro-batch to Postgres"),
        last_batch_rows=Gauge("abs_stream_last_batch_pitches", "Pitches in the latest micro-batch"),
    )
    start_http_server(int(os.environ.get("STREAM_METRICS_PORT", "9109")))

    spark = (SparkSession.builder.appName("abs-stream")
             .master(os.environ.get("SPARK_MASTER", "local[2]"))
             .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3")
             .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "4"))
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    pitch_schema = ddl(PITCH_FIELDS)
    pitches = (spark.readStream.format("kafka")
               .option("kafka.bootstrap.servers", os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092"))
               .option("subscribe", os.environ.get("PITCH_TOPIC", "pitches"))
               .option("startingOffsets", "earliest")
               .option("maxOffsetsPerTrigger", os.environ.get("MAX_OFFSETS_PER_TRIGGER", "5000"))
               .load()
               .select(F.from_json(F.col("value").cast("string"), pitch_schema).alias("p"))
               .select("p.*"))

    scored = pitches.groupBy("game_pk", "chal_team").applyInPandasWithState(
        score_group, outputStructType=ddl(OUT_FIELDS), stateStructType=ddl(STATE_FIELDS),
        outputMode="append", timeoutConf=GroupStateTimeout.NoTimeout)

    query = (scored.writeStream
             .foreachBatch(make_writer(os.environ.get("DATABASE_URL", "postgresql://abs:abs@localhost:5432/abs"),
                                       metrics))
             .option("checkpointLocation", os.environ.get("CHECKPOINT_DIR", "/tmp/abs-stream-checkpoint"))
             .trigger(processingTime=os.environ.get("TRIGGER", "2 seconds"))
             .start())
    query.awaitTermination()


if __name__ == "__main__":
    main()
