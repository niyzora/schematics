"""
FastAPI endpoint tests using TestClient.
Uses a mock model to avoid dependency on a trained model file.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dependencies import get_model, get_preprocessor
from src.inference.preprocessor import Preprocessor
from src.routes import router

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [0.9, 0.5, 0.1]
    return model


@pytest.fixture
def client(mock_model):
    """Create a test app without lifespan, overriding dependencies."""
    test_app = FastAPI()
    test_app.include_router(router)

    real_preprocessor = Preprocessor(DATA_DIR)
    test_app.dependency_overrides[get_preprocessor] = lambda: real_preprocessor
    test_app.dependency_overrides[get_model] = lambda: mock_model

    yield TestClient(test_app)

    test_app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestRankEndpoint:
    def test_rank_returns_sorted_offers(self, client):
        with open(FIXTURES_DIR / "sample_request.json") as f:
            payload = json.load(f)

        resp = client.post("/rank", json=payload)
        assert resp.status_code == 200

        body = resp.json()
        offers = body["ranked_offers"]
        assert len(offers) == 3
        assert offers[0]["rank"] == 1
        assert offers[0]["score"] >= offers[1]["score"]
        assert offers[1]["score"] >= offers[2]["score"]

    def test_rank_empty_offers_rejected(self, client):
        payload = {
            "session": {"traffic_source": "google"},
            "offers": [],
        }
        resp = client.post("/rank", json=payload)
        assert resp.status_code == 422

    def test_rank_missing_brand_rejected(self, client):
        payload = {
            "session": {},
            "offers": [{}],
        }
        resp = client.post("/rank", json=payload)
        assert resp.status_code == 422

    def test_rank_response_has_all_brands(self, client):
        with open(FIXTURES_DIR / "sample_request.json") as f:
            payload = json.load(f)

        resp = client.post("/rank", json=payload)
        brands = {o["brand"] for o in resp.json()["ranked_offers"]}
        expected = {o["brand"] for o in payload["offers"]}
        assert brands == expected
