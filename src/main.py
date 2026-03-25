from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.dependencies import init_resources
from src.routes import router

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_PATH = os.getenv("MODEL_PATH", "data/xgb.json")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")


def setup_telemetry(app: FastAPI) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "heloc-ranker"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:
        logger.warning("OTEL setup failed, running without tracing: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_resources(data_dir=DATA_DIR, model_path=MODEL_PATH)
    yield


app = FastAPI(title="HELOC Offer Ranker", lifespan=lifespan)
app.include_router(router)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

setup_telemetry(app)
