"""
Model wrapper for XGBoost ranker inference.
"""
from __future__ import annotations

from pathlib import Path

import xgboost as xgb

from src.inference.config import FEATURE_COLS


class RankingModel:
    """Loads a trained XGBoost model and scores offer feature vectors."""

    def __init__(self, model_path: str | Path):
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

    def predict(self, feature_matrix: list[list[float]]) -> list[float]:
        """
        Score each row in the feature matrix.
        Returns a list of float scores (higher = better rank).
        """
        dmat = xgb.DMatrix(feature_matrix, feature_names=FEATURE_COLS)
        return self.model.predict(dmat).tolist()

    @property
    def expected_features(self) -> list[str]:
        return FEATURE_COLS
