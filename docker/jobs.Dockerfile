# Spark streaming job, Redpanda producer, MLflow server and trainer, drift monitor.
# Debian bookworm still ships Java 17, which Spark 3.5 needs.
FROM python:3.11-slim-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/jobs.txt pytest httpx
COPY abslive/ abslive/
COPY stream/ stream/
COPY model/ model/
COPY data/ data/
COPY tests/ tests/
COPY pytest.ini ./
# Fetch the Spark Kafka connector now, so the job starts without network access.
RUN python -c "from pyspark.sql import SparkSession; SparkSession.builder.master('local[1]').config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3').getOrCreate().stop()"
ENV PYTHONUNBUFFERED=1
