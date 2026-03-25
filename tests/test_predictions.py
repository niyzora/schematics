"""
Tests that model predictions produce expected scores.
These tests will pass only after a trained model is saved to data/cb_ranker.cbm.
"""
import pytest
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "data" / "xgb.json"
model_available = MODEL_PATH.exists()


@pytest.mark.skipif(not model_available, reason="No trained model at data/cb_ranker.cbm")
class TestPredictions:
    @pytest.fixture(autouse=True)
    def _load_model(self, preprocessor):
        from src.inference.model import RankingModel
        self.model = RankingModel(MODEL_PATH)
        self.preprocessor = preprocessor

    def test_scores_are_finite(self, sample_request):
        matrix = self.preprocessor.to_feature_matrix(
            sample_request["session"],
            sample_request["offers"],
        )
        scores = self.model.predict(matrix)
        assert len(scores) == len(sample_request["offers"])
        assert all(isinstance(s, float) for s in scores)
        assert all(s == s for s in scores)  # no NaN

    def test_high_ev_brand_scores_higher(self):
        """A brand with higher historical EV should get a higher score."""
        session = {"credit_score_rate": "Good", "traffic_source": "google"}
        offers_high = [{"brand": "amerisave"}]  # high EV brand
        offers_low = [{"brand": "amone"}]  # low EV brand

        score_high = self.model.predict(
            self.preprocessor.to_feature_matrix(session, offers_high)
        )[0]
        score_low = self.model.predict(
            self.preprocessor.to_feature_matrix(session, offers_low)
        )[0]
        assert score_high > score_low

    def test_different_brands_produce_different_scores(self, sample_request):
        """With multiple brands, scores should not all be identical."""
        matrix = self.preprocessor.to_feature_matrix(
            sample_request["session"],
            sample_request["offers"],
        )
        scores = self.model.predict(matrix)
        assert len(set(scores)) > 1
