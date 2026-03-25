import json
from pathlib import Path

import pytest

from src.inference.preprocessor import Preprocessor

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def preprocessor():
    return Preprocessor(DATA_DIR)


@pytest.fixture
def sample_request():
    with open(FIXTURES_DIR / "sample_request.json") as f:
        return json.load(f)


@pytest.fixture
def expected_features():
    with open(FIXTURES_DIR / "expected_features.json") as f:
        return json.load(f)
