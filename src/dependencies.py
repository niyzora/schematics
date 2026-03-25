"""
Singleton holders for the preprocessor and model, loaded once at startup.
"""
from __future__ import annotations

from src.inference.model import RankingModel
from src.inference.preprocessor import Preprocessor

_preprocessor: Preprocessor | None = None
_model: RankingModel | None = None


def init_resources(data_dir: str, model_path: str) -> None:
    global _preprocessor, _model
    _preprocessor = Preprocessor(data_dir)
    _model = RankingModel(model_path)


def get_preprocessor() -> Preprocessor:
    assert _preprocessor is not None, "Preprocessor not initialized"
    return _preprocessor


def get_model() -> RankingModel:
    assert _model is not None, "Model not initialized"
    return _model
