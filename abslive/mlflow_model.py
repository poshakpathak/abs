"""The model as an MLflow pyfunc: a table of situations in, decisions out."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import mlflow.pyfunc

    _Base = mlflow.pyfunc.PythonModel
except ImportError:  # the library itself does not need MLflow
    _Base = object


class ChallengePolicy(_Base):
    def load_context(self, context):
        from .policy import Tables

        self.tables = Tables.load(Path(context.artifacts["model_dir"]))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from .stake_input import Situation, evaluate

        rows = [evaluate(self.tables, Situation(**r)) for r in model_input.to_dict(orient="records")]
        return pd.DataFrame(rows)
