"""FastAPI application: health, trip listing, observations, and prediction."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import pathsnap
from api import deps, schemas
from api.service import PredictionService
from pathsnap import config

logger = logging.getLogger("pathsnap.api")


def _require_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Prediction service is not ready")
    return service


def create_app(service: PredictionService | None = None) -> FastAPI:
    """Build the app; tests may inject a preloaded service."""
    logging.basicConfig(
        level=deps.log_level(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "service", None) is None:
            paths = config.get_paths()
            app.state.service = PredictionService(
                paths.eval_data_dir, paths.artifact_dir
            )
        yield

    app = FastAPI(
        title="PathSnap API",
        version=pathsnap.__version__,
        lifespan=lifespan,
    )
    if service is not None:
        app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=deps.cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "request_id=%s method=%s path=%s status=500 latency_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response

    @app.get("/health", response_model=schemas.HealthResponse)
    def health(request: Request) -> schemas.HealthResponse:
        service = getattr(request.app.state, "service", None)
        return schemas.HealthResponse(
            status="ok" if service is not None else "starting",
            model_loaded=service is not None,
            version=pathsnap.__version__,
            model_name=service.model_name if service is not None else None,
        )

    @app.get("/trips", response_model=schemas.TripsResponse)
    def list_trips(request: Request) -> schemas.TripsResponse:
        service = _require_service(request)
        return schemas.TripsResponse(trips=service.list_trips())

    @app.get(
        "/trips/{trip_id}/observations",
        response_model=schemas.ObservationsResponse,
    )
    def observations(trip_id: str, request: Request) -> schemas.ObservationsResponse:
        service = _require_service(request)
        try:
            points = service.observations(trip_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Trip not found") from None
        return schemas.ObservationsResponse(trip_id=trip_id, points=points)

    @app.post("/predict", response_model=schemas.PredictResponse)
    def predict(payload: schemas.PredictRequest, request: Request) -> schemas.PredictResponse:
        service = _require_service(request)
        try:
            response = service.predict(
                trip_id=payload.trip_id,
                method=payload.method,
                noise_sigma_m=payload.noise_sigma_m,
                sample_every_s=payload.sample_every_s,
                seed=payload.seed,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Trip not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

        logger.info(
            "prediction request_id=%s trip_id=%s method=%s inference_ms=%.1f "
            "top1_accuracy=%.4f route_continuity=%.4f",
            getattr(request.state, "request_id", "-"),
            payload.trip_id,
            payload.method,
            response.inference_ms,
            response.metrics.top1_accuracy,
            response.metrics.route_continuity,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers={"X-Request-ID": getattr(request.state, "request_id", "")},
        )

    # The UI and API share an origin when launched with the included launcher.
    ui_dir = config.get_paths().project_root / "ui"
    if ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")
    return app


app = create_app()
