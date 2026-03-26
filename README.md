# HELOC Offer Ranking Service

Home assignment: prioritize HELOC (Home Equity Line of Credit) offers for users using. 

## Project Structure

```
schematics_eda.ipynb       — MAIN FILE: EDA + Feature Engineering + Model training
src/                       — FastAPI ranking service
  main.py                  — App entrypoint, OTEL/Prometheus setup
  routes.py                — POST /rank, GET /health, GET /metrics
  schemas.py               — Request/response Pydantic models
  inference/
    config.py              — Feature columns, ordinal mappings
    preprocessor.py        — Real-time feature engineering using JSON lookups
    model.py               — XGB model wrapper
deployment/
  Dockerfile
  deployment.yaml          — K8s deployment 
  service.yaml             — K8s service
  hpa.yaml                 — Horizontal Pod Autoscaler for autoscaling
tests/
  test_preprocessor.py     — Feature engineering unit tests
  test_predictions.py      — Model output validation
  test_service.py          — API endpoint tests
  fixtures/                — Sample payloads and expected outputs
data/
  xgb.json            — Trained XGB ranker
  brand_lookup.json        — Brand-level aggregate statistics
  brand_credit_lookup.json — Brand × credit score interaction features
  traffic_lookup.json      — Traffic source statistics
  imputation_params.json   — Median values for missing feature imputation
```

## EDA Key Findings

- Position bias: position 1 receives ~50% CTR, others disproportionally lower. This means clicks are influenced by display order, not just offer quality
- 98% of sessions have zero or one click; max clicks per session is 4
- Duplicate brands per session exist (varying conversion timestamps)
- Multiple clicks with different payouts per session suggest several levels of relevance
- 8 long-tail brands contribute less than 1% of traffic
- Some brands like "lendingtree" have low avg position (3.13) but top CTR at position 1; others like "figure" show the opposite pattern
- Global trend: wealthier profiles have lower CTR
- Some brands target premium users with higher payouts (e.g. "Figure"), others target average segments (e.g. "lendingtree")
- Most brands target Good/Fair credit score segments; expected value drops significantly for the "Excellent" segment

## Modelling

Final tuned XGboost ranker trained with ndcg objective. Tried both pairwise and listwise methods, but during tuning listwise showed better results. 

Relevance labels (final): 1 if clicked and 1+log1p(payouts) if converted, that helped to take into account both click and payout.
 
At inference, position features are fixed to 1 (deconfounded) as one method of coping with position bias (we trained using it). However in theory different methods could be explored like Position-bias aware learning framework, Inverse Propensity Weights etc.

## Inference Pipeline

The preprocessing in `src/inference/preprocessor.py` mirrors the notebook feature engineering:
1. Ordinal encoding of user features (income, credit score, credit line, property value, mortgage)
2. Missing value flags + median imputation
3. One-hot encoding (loan purpose, device, property type/use)
4. Brand aggregate lookup (smoothed CTR, EV, position-1 CTR)
5. Brand × credit score interaction lookup with hierarchical fallback
6. Traffic source lookup
7. Position features fixed to 1

All lookups are precomputed JSON files for lightweight inference.

## Deployment

FastAPI service deployed to Kubernetes:
- **Framework**: FastAPI (async, fast, typed)
- **Monitoring**: OTEL collector sidecar for traces + Prometheus for metrics (latency histograms, request counts). 
- **Scaling**: HPA based on CPU/memory utilization (2–10 replicas)

Deployment workflow:
1. Save model and lookup artifacts to cloud storage (e.g. S3)
2. Run local tests (`pytest tests/`)
3. Build Docker image, push to ECR
4. Deploy to dev/preprod, validate
5. Deploy to prod

## Running

```bash
# Tests
pip install -r requirements.txt
pytest tests/ -v

# Local service
uvicorn src.main:app --reload
```

