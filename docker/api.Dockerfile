# The model API. This is the image CI publishes to GHCR.
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY requirements/core.txt requirements/api.txt requirements/
RUN pip install --no-cache-dir -r requirements/api.txt
COPY abslive/ abslive/
COPY service/ service/
# The paper's tables, served when no MLflow registry is configured.
COPY model/ model/
RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
