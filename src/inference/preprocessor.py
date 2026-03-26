"""
Real-time feature preprocessing for the HELOC ranking service.
Mirrors the feature engineering pipeline from the training notebook
using precomputed JSON lookup tables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.inference.config import (
    ANNUAL_INCOME_MAP,
    CREDIT_LINE_MAP,
    CREDIT_SCORE_MAP,
    DEVICE_DUMMIES,
    FEATURE_COLS,
    LOAN_PURPOSE_DUMMIES,
    MORTGAGE_MAP,
    PROPERTY_TYPE_DUMMIES,
    PROPERTY_USE_DUMMIES,
    PROPERTY_VALUE_MAP,
)


class Preprocessor:
    """Transforms raw session + offer data into a feature matrix for the ranker."""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.brand_lookup = self._load_json(data_dir / "brand_lookup.json")
        self.brand_credit_lookup = self._load_json(data_dir / "brand_credit_lookup.json")
        self.traffic_lookup = self._load_json(data_dir / "traffic_lookup.json")
        self.imputation = self._load_json(data_dir / "imputation_params.json")

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def transform(
        self,
        session: dict[str, Any],
        offers: list[dict[str, Any]],
    ) -> list[dict[str, float]]:
        """
        Build one feature dict per offer, returned in the same order as `offers`.
        Each dict has keys matching FEATURE_COLS.
        """
        user = self._encode_user(session)
        return [self._encode_offer(user, offer) for offer in offers]

    def _encode_user(self, session: dict[str, Any]) -> dict[str, float]:
        """Encode session-level (user) features once per request."""
        features: dict[str, float] = {}

        # ordinal encoding
        credit_score_ord = CREDIT_SCORE_MAP.get(session.get("credit_score_rate"))
        income_ord = ANNUAL_INCOME_MAP.get(session.get("annual_income"))
        credit_line_ord = CREDIT_LINE_MAP.get(session.get("credit_line"))
        property_value_ord = PROPERTY_VALUE_MAP.get(session.get("property_value"))
        mortgage_ord = MORTGAGE_MAP.get(session.get("currently_have_mortgage"))

        # missing flags
        features["annual_income_missing"] = float(income_ord is None)
        features["credit_line_missing"] = float(credit_line_ord is None)
        features["military_missing"] = float(session.get("military_veteran") is None)

        # imputation with train medians
        features["credit_score_ord"] = float(
            credit_score_ord if credit_score_ord is not None
            else self.imputation["credit_score_median"]
        )
        features["annual_income_ord"] = float(
            income_ord if income_ord is not None
            else self.imputation["income_median"]
        )
        features["credit_line_ord"] = float(
            credit_line_ord if credit_line_ord is not None
            else self.imputation["cline_median"]
        )
        features["property_value_ord"] = float(
            property_value_ord if property_value_ord is not None
            else self.imputation["property_value_median"]
        )
        features["mortgage_ord"] = float(mortgage_ord if mortgage_ord is not None else 1)

        # derived 
        features["is_military"] = float(session.get("military_veteran") == "Yes")

        # one-hot: loan purpose 
        purpose = session.get("loan_primary_purpose", "")
        for cat in LOAN_PURPOSE_DUMMIES:
            features[f"loan_purpose_{cat}"] = float(purpose == cat)

        # one-hot: device 
        device = session.get("device_type", "")
        for cat in DEVICE_DUMMIES:
            features[f"device_{cat}"] = float(device == cat)

        # one-hot: property_type 
        ptype = session.get("property_type", "")
        for cat in PROPERTY_TYPE_DUMMIES:
            features[f"property_type_{cat}"] = float(ptype == cat)

        # one-hot: property_use 
        puse = session.get("property_use", "")
        for cat in PROPERTY_USE_DUMMIES:
            features[f"property_use_{cat}"] = float(puse == cat)

        # traffic source lookup 
        ts = session.get("traffic_source", "")
        ts_stats = self.traffic_lookup.get(ts, {})
        features["traffic_ctr"] = ts_stats.get("traffic_ctr", self.imputation["global_ctr"])
        features["traffic_ev"] = ts_stats.get("traffic_ev", 0.0)

        return features

    def _encode_offer(
        self,
        user_features: dict[str, float],
        offer: dict[str, Any],
    ) -> dict[str, float]:
        """Combine user features with per-offer (brand + position) features."""
        features = dict(user_features)

        brand = offer["brand"]

        # At inference all offers are scored at position=1 (deconfounded).
        # The model was trained with position bias; fixing position=1 ensures
        # ranking is based on intrinsic brand×user affinity, not display order.
        features["position"] = 1.0
        features["position_inv"] = 1.0
        features["is_position_1"] = 1.0

        # brand lookup
        b_stats = self.brand_lookup.get(brand, {})
        global_ctr = self.imputation["global_ctr"]
        features["brand_ctr"] = b_stats.get("brand_ctr", global_ctr)
        features["brand_ev"] = b_stats.get("brand_ev", 0.0)
        features["brand_pos1_ctr"] = b_stats.get("brand_pos1_ctr", global_ctr)

        # brand × credit_score lookup
        credit_key = f"{brand}|{int(features['credit_score_ord'])}"
        bc_stats = self.brand_credit_lookup.get(credit_key, {})
        features["brand_credit_ctr"] = bc_stats.get("brand_credit_ctr", features["brand_ctr"])
        features["brand_credit_lift"] = bc_stats.get("brand_credit_lift", 1.0)

        return features

    def to_feature_matrix(
        self,
        session: dict[str, Any],
        offers: list[dict[str, Any]],
    ) -> list[list[float]]:
        """Return a 2D list (n_offers × n_features) in FEATURE_COLS order."""
        rows = self.transform(session, offers)
        return [[row[col] for col in FEATURE_COLS] for row in rows]
