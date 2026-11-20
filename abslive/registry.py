"""Find the model to serve: the MLflow champion if a registry is configured, else the files in model/."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .policy import Tables

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def resolve(model_uri: str | None = None, model_dir: str | None = None) -> tuple[Path, str]:
    """Return (local directory with the model files, version label).

    MODEL_URI like `models:/abs-challenge-policy@champion` pulls from the MLflow
    registry named by MLFLOW_TRACKING_URI. Otherwise MODEL_DIR (default model/)
    is used as-is with the version label "research".
    """
    model_uri = model_uri if model_uri is not None else os.environ.get("MODEL_URI", "")
    if model_uri:
        import mlflow
        from mlflow import MlflowClient

        version = model_uri
        if model_uri.startswith("models:/") and "@" in model_uri:
            name, alias = model_uri[len("models:/"):].split("@", 1)
            mv = MlflowClient().get_model_version_by_alias(name, alias)
            model_uri = f"models:/{name}/{mv.version}"
            version = f"{name}/v{mv.version}"
        local = Path(mlflow.artifacts.download_artifacts(model_uri))
        found = list(local.rglob("research_tables.npz"))
        if not found:
            raise FileNotFoundError(f"no model files in {model_uri}")
        log.info("loaded %s from %s", version, found[0].parent)
        return found[0].parent, version
    directory = Path(model_dir or os.environ.get("MODEL_DIR", ROOT / "model"))
    return directory, "research"


def load_tables(model_uri: str | None = None, model_dir: str | None = None) -> tuple[Tables, Path]:
    directory, version = resolve(model_uri, model_dir)
    return Tables.load(directory, version=version), directory
