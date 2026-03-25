import pytest

from src.inference.config import FEATURE_COLS


class TestPreprocessor:
    def test_output_has_all_feature_cols(self, preprocessor, sample_request):
        rows = preprocessor.transform(
            sample_request["session"],
            sample_request["offers"],
        )
        assert len(rows) == len(sample_request["offers"])
        for row in rows:
            assert set(row.keys()) == set(FEATURE_COLS)

    def test_feature_values_match_expected(self, preprocessor, sample_request, expected_features):
        rows = preprocessor.transform(
            sample_request["session"],
            sample_request["offers"],
        )
        for row, expected in zip(rows, expected_features["offers"]):
            for key, expected_val in expected["features"].items():
                actual = row[key]
                assert actual == pytest.approx(expected_val, abs=1e-6), (
                    f"{expected['brand']}.{key}: got {actual}, expected {expected_val}"
                )

    def test_missing_income_flag(self, preprocessor):
        session = {"credit_score_rate": "Good"}
        offers = [{"brand": "achieve"}]
        row = preprocessor.transform(session, offers)[0]
        assert row["annual_income_missing"] == 1.0
        assert row["annual_income_ord"] == 3.0  # imputed median

    def test_missing_military_flag(self, preprocessor):
        session = {}
        offers = [{"brand": "achieve"}]
        row = preprocessor.transform(session, offers)[0]
        assert row["military_missing"] == 1.0
        assert row["is_military"] == 0.0

    def test_unknown_brand_uses_global_fallback(self, preprocessor):
        session = {"credit_score_rate": "Good"}
        offers = [{"brand": "nonexistent_brand", "position": 1}]
        row = preprocessor.transform(session, offers)[0]
        assert row["brand_ctr"] == pytest.approx(0.2104, abs=0.01)
        assert row["brand_ev"] == 0.0
        assert row["brand_credit_ctr"] == row["brand_ctr"]
        assert row["brand_credit_lift"] == 1.0

    def test_unknown_traffic_source_fallback(self, preprocessor):
        session = {"traffic_source": "snapchat"}
        offers = [{"brand": "achieve"}]
        row = preprocessor.transform(session, offers)[0]
        assert row["traffic_ctr"] == pytest.approx(0.2104, abs=0.01)
        assert row["traffic_ev"] == 0.0

    def test_position_always_deconfounded(self, preprocessor):
        """All offers scored at position=1 regardless of input."""
        session = {}
        offers = [{"brand": "achieve"}, {"brand": "figure"}]
        rows = preprocessor.transform(session, offers)
        for row in rows:
            assert row["position"] == 1.0
            assert row["position_inv"] == 1.0
            assert row["is_position_1"] == 1.0

    def test_one_hot_loan_purpose(self, preprocessor):
        session = {"loan_primary_purpose": "Home Improvements"}
        offers = [{"brand": "achieve"}]
        row = preprocessor.transform(session, offers)[0]
        assert row["loan_purpose_Home Improvements"] == 1.0
        assert row["loan_purpose_Other"] == 0.0

    def test_one_hot_base_category_is_all_zeros(self, preprocessor):
        session = {"loan_primary_purpose": "Debt Consolidation", "device_type": "desktop"}
        offers = [{"brand": "achieve"}]
        row = preprocessor.transform(session, offers)[0]
        assert row["loan_purpose_Home Improvements"] == 0.0
        assert row["loan_purpose_Investment Opportunities"] == 0.0
        assert row["loan_purpose_Other"] == 0.0
        assert row["loan_purpose_Retirement Income"] == 0.0
        assert row["device_mobile"] == 0.0
        assert row["device_tablet"] == 0.0

    def test_to_feature_matrix_shape(self, preprocessor, sample_request):
        matrix = preprocessor.to_feature_matrix(
            sample_request["session"],
            sample_request["offers"],
        )
        assert len(matrix) == 3
        assert all(len(row) == len(FEATURE_COLS) for row in matrix)

    def test_brand_credit_cross_feature(self, preprocessor):
        session = {"credit_score_rate": "Good"}  # maps to 4
        offers = [{"brand": "achieve"}]
        row = preprocessor.transform(session, offers)[0]
        # achieve|4 exists in lookup → should use it, not fallback
        assert row["brand_credit_ctr"] != row["brand_ctr"]
