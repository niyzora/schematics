from __future__ import annotations

from pydantic import BaseModel, Field


class SessionContext(BaseModel):
    traffic_source: str | None = None
    device_type: str | None = None
    credit_score_rate: str | None = None
    annual_income: str | None = None
    credit_line: str | None = None
    property_value: str | None = None
    property_type: str | None = None
    property_use: str | None = None
    currently_have_mortgage: str | None = None
    military_veteran: str | None = None
    loan_primary_purpose: str | None = None


class Offer(BaseModel):
    brand: str


class RankRequest(BaseModel):
    session: SessionContext
    offers: list[Offer] = Field(min_length=1)


class RankedOffer(BaseModel):
    brand: str
    score: float
    rank: int


class RankResponse(BaseModel):
    ranked_offers: list[RankedOffer]
