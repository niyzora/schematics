from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from prometheus_client import Histogram

from src.dependencies import get_model, get_preprocessor
from src.inference.model import RankingModel
from src.inference.preprocessor import Preprocessor
from src.schemas import RankedOffer, RankRequest, RankResponse

router = APIRouter()

PREDICTION_LATENCY = Histogram(
    "ranking_prediction_seconds",
    "Time spent computing ranking predictions",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/rank", response_model=RankResponse)
def rank(
    request: RankRequest,
    preprocessor: Preprocessor = Depends(get_preprocessor),
    model: RankingModel = Depends(get_model),
):
    session_dict = request.session.model_dump()
    offers_dict = [o.model_dump() for o in request.offers]

    start = time.perf_counter()
    feature_matrix = preprocessor.to_feature_matrix(session_dict, offers_dict)
    scores = model.predict(feature_matrix)
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    scored = sorted(
        zip(request.offers, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    ranked_offers = [
        RankedOffer(brand=offer.brand, score=round(score, 6), rank=i + 1)
        for i, (offer, score) in enumerate(scored)
    ]
    return RankResponse(ranked_offers=ranked_offers)
