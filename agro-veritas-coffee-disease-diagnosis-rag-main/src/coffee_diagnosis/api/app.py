"""FastAPI application for CoffeeGuard diagnosis and incident support."""

import json
import logging
import time
from typing import Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from coffee_diagnosis.api.metrics import (
    get_labeled_evaluation,
    HTTP_ERRORS,
    HTTP_REQUESTS,
    HTTP_REQUEST_DURATION,
    load_labeled_evaluation,
    render_metrics,
)
from coffee_diagnosis.api.slack import (
    SlackIntegration,
    SlackNotificationError,
    SlackRequestError,
)
from coffee_diagnosis.api.incident_monitor import IncidentMonitor
from coffee_diagnosis.api.schemas import (
    AnswerRequest,
    DiagnosisRequest,
    IncidentCreateRequest,
    IncidentRecord,
    IncidentResolutionRequest,
    InvestigationRequest,
    KnowledgeSearchRequest,
)
from coffee_diagnosis.api.service import (
    DiagnosisService,
    IncidentNotFoundError,
    IncidentService,
    IncidentStateConflictError,
    InvalidSessionStateError,
    SessionCapacityError,
    SessionNotFoundError,
)
from coffee_diagnosis.api.support_agent import SupportAgent
from config import settings


class JsonLogFormatter(logging.Formatter):
    """Emit compact structured logs without request body or symptom content."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status_code", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=True)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("coffee_guard.api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.API_LOG_LEVEL, logging.INFO))
    logger.propagate = False
    return logger


logger = _configure_logging()


def create_app(
    diagnosis_service: Optional[DiagnosisService] = None,
    incident_service: Optional[IncidentService] = None,
) -> FastAPI:
    application = FastAPI(
        title="CoffeeGuard API",
        version="1.0.0",
        description="Coffee disease diagnosis and operational incident support API.",
    )
    application.state.diagnosis_service = diagnosis_service or DiagnosisService()
    application.state.incident_service = incident_service or IncidentService()
    application.state.incident_monitor = IncidentMonitor(
        application.state.incident_service
    )
    application.state.support_agent = SupportAgent(
        application.state.diagnosis_service,
        application.state.incident_service,
        application.state.incident_monitor,
    )
    application.state.slack = SlackIntegration(
        application.state.support_agent,
        application.state.diagnosis_service,
    )
    application.state.evaluation_metadata = load_labeled_evaluation(
        settings.DIAGNOSIS_EVALUATION_FILE
    )

    @application.middleware("http")
    async def observe_request(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        if len(request_id) > 128 or any(ord(char) < 32 for char in request_id):
            request_id = str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500
        content_length = request.headers.get("content-length", "0")
        if content_length.isdigit() and int(content_length) > settings.API_MAX_REQUEST_BYTES:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "request_too_large",
                        "message": "Request body exceeds the configured size limit.",
                        "request_id": request_id,
                    }
                },
            )
        else:
            response = await call_next(request)
        status_code = response.status_code
        route = request.scope.get("route")
        metric_path = getattr(route, "path", request.url.path)
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, metric_path, str(status_code)).inc()
        HTTP_REQUEST_DURATION.labels(request.method, metric_path).observe(elapsed)
        if status_code >= 400:
            HTTP_ERRORS.labels(request.method, metric_path, str(status_code)).inc()
        if metric_path != "/metrics":
            new_incidents = application.state.incident_monitor.observe(
                request.method,
                metric_path,
                status_code,
                elapsed,
                request_id,
            )
            for incident in new_incidents:
                try:
                    triage = await run_in_threadpool(
                        application.state.support_agent.investigate,
                        incident["title"],
                    )
                    incident_record = application.state.incident_service.add_triage(
                        UUID(incident["id"]),
                        {
                            "triage_status": "complete",
                            "triage_summary": triage["summary"],
                            "triage_sources": [
                                evidence["source"]
                                for evidence in triage["evidence"]
                            ],
                        },
                    )
                    incident = incident_record.model_dump(mode="json")
                except Exception as exc:
                    logger.warning(
                        "incident_triage_failed",
                        extra={"error_type": type(exc).__name__},
                    )
                    application.state.incident_service.add_triage(
                        UUID(incident["id"]),
                        {
                            "triage_status": "unavailable",
                            "triage_error": type(exc).__name__,
                        },
                    )
                try:
                    await application.state.slack.notify_incident(incident)
                except SlackNotificationError as exc:
                    logger.warning(
                        "incident_notification_failed",
                        extra={"error_type": type(exc).__name__},
                    )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": metric_path,
                "status_code": status_code,
                "duration_ms": round(elapsed * 1000, 2),
            },
        )
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        issues = [
            {"location": list(error["loc"]), "message": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": issues,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        code = {
            404: "not_found",
            413: "request_too_large",
        }.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
            headers=exc.headers,
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.exception(
            "unhandled_request_error",
            extra={"request_id": getattr(request.state, "request_id", None)},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "The request could not be completed.",
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @application.get("/health/live", tags=["health"])
    async def live():
        return {"status": "alive"}

    @application.get("/health/ready", tags=["health"])
    async def ready():
        missing = []
        if not settings.PDF_DIR.is_dir() or not any(settings.PDF_DIR.glob("*.pdf")):
            missing.append("PDF knowledge files")
        disease_json = settings.DATA_DIR / "disease_knowledge.json"
        if not disease_json.is_file():
            missing.append("structured disease knowledge")
        if missing:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "missing": missing},
            )
        return {
            "status": "ready",
            "controller_initialized": (
                application.state.diagnosis_service._controller is not None
            ),
        }

    @application.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    @application.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(render_metrics(), media_type=CONTENT_TYPE_LATEST)

    @application.get("/api/v1/telemetry", tags=["observability"])
    async def telemetry():
        return application.state.incident_monitor.telemetry_snapshot()

    @application.get("/api/v1/errors/recent", tags=["observability"])
    async def recent_errors(limit: int = Query(default=50, ge=1, le=200)):
        return {
            "source": "measured_request_telemetry",
            "simulated": False,
            "errors": application.state.incident_monitor.recent_errors(limit),
        }

    @application.post("/api/v1/knowledge/search", tags=["support"])
    def search_knowledge(payload: KnowledgeSearchRequest):
        return application.state.support_agent.knowledge_search(
            payload.query, payload.limit
        )

    @application.post("/api/v1/support/investigate", tags=["support"])
    def investigate(payload: InvestigationRequest):
        return application.state.support_agent.investigate(payload.query)

    @application.post("/api/v1/slack/commands", tags=["slack"])
    async def slack_commands(request: Request):
        if not application.state.slack.signing_secret:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "slack_not_configured",
                        "message": "Slack commands are disabled until signing credentials are configured.",
                    }
                },
            )
        raw_body = await request.body()
        if len(raw_body) > settings.API_MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "request_too_large",
                        "message": "Request body exceeds the configured size limit.",
                    }
                },
            )
        try:
            response = await run_in_threadpool(
                application.state.slack.handle_command,
                raw_body,
                request.headers.get("X-Slack-Request-Timestamp", ""),
                request.headers.get("X-Slack-Signature", ""),
            )
        except SlackRequestError as exc:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "invalid_slack_request",
                        "message": str(exc),
                    }
                },
            )
        return response

    @application.get("/api/v1/quality/evaluation", tags=["observability"])
    async def evaluation_quality():
        evaluation = get_labeled_evaluation()
        if evaluation is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "unavailable",
                    "message": (
                        "No labeled evaluation results are configured. "
                        "Live diagnosis predictions are not used for quality metrics."
                    ),
                },
            )
        return evaluation

    @application.post("/api/v1/diagnosis", tags=["diagnosis"])
    def start_diagnosis(payload: DiagnosisRequest):
        try:
            result = application.state.diagnosis_service.start_diagnosis(payload.query)
        except SessionCapacityError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "session_capacity",
                        "message": str(exc),
                    }
                },
            )
        return jsonable_encoder(result)

    @application.post(
        "/api/v1/diagnosis/{session_id}/answers", tags=["diagnosis"]
    )
    def submit_answer(session_id: str, payload: AnswerRequest):
        try:
            UUID(session_id)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "invalid_session_id",
                        "message": "session_id must be a UUID.",
                    }
                },
            )
        try:
            result = application.state.diagnosis_service.submit_answer(
                session_id, payload.answer
            )
        except SessionNotFoundError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "session_not_found",
                        "message": "Diagnosis session does not exist or has expired.",
                    }
                },
            )
        except InvalidSessionStateError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": {"code": "invalid_session_state", "message": str(exc)}},
            )
        return jsonable_encoder(result)

    @application.delete(
        "/api/v1/diagnosis/{session_id}", status_code=204, tags=["diagnosis"]
    )
    def delete_session(session_id: str):
        try:
            UUID(session_id)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "invalid_session_id",
                        "message": "session_id must be a UUID.",
                    }
                },
            )
        try:
            application.state.diagnosis_service.delete_session(session_id)
        except SessionNotFoundError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "session_not_found",
                        "message": "Diagnosis session does not exist or has expired.",
                    }
                },
            )
        return Response(status_code=204)

    @application.get("/api/v1/incidents", response_model=list[IncidentRecord], tags=["incidents"])
    def list_incidents():
        return application.state.incident_service.list()

    @application.post(
        "/api/v1/incidents",
        response_model=IncidentRecord,
        status_code=201,
        tags=["incidents"],
    )
    async def create_incident(payload: IncidentCreateRequest):
        incident = application.state.incident_service.create(payload)
        try:
            triage = await run_in_threadpool(
                application.state.support_agent.investigate,
                incident.title,
            )
            incident = application.state.incident_service.add_triage(
                incident.id,
                {
                    "triage_status": "complete",
                    "triage_summary": triage["summary"],
                    "triage_sources": [
                        evidence["source"] for evidence in triage["evidence"]
                    ],
                },
            )
        except Exception as exc:
            logger.warning(
                "incident_triage_failed",
                extra={"error_type": type(exc).__name__},
            )
            incident = application.state.incident_service.add_triage(
                incident.id,
                {
                    "triage_status": "unavailable",
                    "triage_error": type(exc).__name__,
                },
            )
        try:
            await application.state.slack.notify_incident(
                incident.model_dump(mode="json")
            )
        except SlackNotificationError as exc:
            logger.warning(
                "incident_notification_failed",
                extra={"error_type": type(exc).__name__},
            )
        return incident

    @application.get(
        "/api/v1/incidents/{incident_id}",
        response_model=IncidentRecord,
        tags=["incidents"],
    )
    def get_incident(incident_id: UUID):
        try:
            return application.state.incident_service.get(incident_id)
        except IncidentNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "incident_not_found", "message": "Incident not found."}},
            )

    @application.post(
        "/api/v1/incidents/{incident_id}/acknowledge",
        response_model=IncidentRecord,
        tags=["incidents"],
    )
    def acknowledge_incident(incident_id: UUID):
        try:
            return application.state.incident_service.acknowledge(incident_id)
        except IncidentNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "incident_not_found", "message": "Incident not found."}},
            )
        except IncidentStateConflictError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": {"code": "incident_state_conflict", "message": str(exc)}},
            )

    @application.post(
        "/api/v1/incidents/{incident_id}/resolve",
        response_model=IncidentRecord,
        tags=["incidents"],
    )
    def resolve_incident(
        incident_id: UUID, payload: IncidentResolutionRequest
    ):
        try:
            return application.state.incident_service.resolve(
                incident_id, payload.resolution
            )
        except IncidentNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "incident_not_found", "message": "Incident not found."}},
            )
        except IncidentStateConflictError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": {"code": "incident_state_conflict", "message": str(exc)}},
            )

    return application


app = create_app()
