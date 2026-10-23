"""Isolated multi-turn diagnosis sessions and in-memory incident workflows."""

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import logging
import sqlite3
from threading import RLock
from typing import Callable, Dict, Optional
from uuid import UUID, uuid4

from config import settings
from coffee_diagnosis.api.metrics import (
    DIAGNOSIS_REQUESTS,
    INCIDENTS,
    RETRIEVAL_OUTCOMES,
)
from coffee_diagnosis.api.schemas import IncidentCreateRequest, IncidentRecord
from coffee_diagnosis.rag.state_manager import StateManager

logger = logging.getLogger("coffee_guard.api.service")


class SessionNotFoundError(KeyError):
    """Raised when a multi-turn session is unknown or expired."""


class InvalidSessionStateError(ValueError):
    """Raised when an answer is submitted before a diagnosis session starts."""


class IncidentNotFoundError(KeyError):
    """Raised when an incident identifier is unknown."""


class IncidentStateConflictError(ValueError):
    """Raised when an incident status transition is not allowed."""


class SessionCapacityError(RuntimeError):
    """Raised when the service has reached its configured session limit."""


class KnowledgeUnavailableError(RuntimeError):
    """Raised when disease-document retrieval cannot be served."""


class DiagnosisService:
    """Serializes access to the shared retrieval engine while isolating user state."""

    def __init__(
        self,
        controller_factory: Optional[Callable[[], object]] = None,
        max_sessions: Optional[int] = None,
        session_ttl_seconds: Optional[int] = None,
    ):
        self._controller_factory = controller_factory or self._default_controller_factory
        self._controller = None
        self._sessions: OrderedDict[str, Dict] = OrderedDict()
        self._max_sessions = (
            settings.API_MAX_SESSIONS if max_sessions is None else max_sessions
        )
        self._session_ttl = (
            settings.API_SESSION_TTL_SECONDS
            if session_ttl_seconds is None
            else session_ttl_seconds
        )
        if self._max_sessions < 1 or self._session_ttl < 1:
            raise ValueError("max_sessions and session_ttl_seconds must be positive.")
        self._lock = RLock()

    @staticmethod
    def _default_controller_factory():
        from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController

        return CoffeeDiagnosisController(
            data_dir=str(settings.PDF_DIR),
            vector_db_path=str(settings.VECTOR_DB_PATH),
        )

    def _get_controller(self):
        if self._controller is None:
            self._controller = self._controller_factory()
        return self._controller

    def _prune_expired(self, now: datetime) -> None:
        expiration = now - timedelta(seconds=self._session_ttl)
        for session_id in list(self._sessions):
            if self._sessions[session_id]["updated_at"] < expiration:
                del self._sessions[session_id]

    def _activate_session(self, controller, session: Dict) -> None:
        controller.state_manager = deepcopy(session["state_manager"])
        controller.clarification_gen.asked_priority_questions = set(
            session["asked_priority_questions"]
        )

    def _save_session(self, session: Dict, controller) -> None:
        session["state_manager"] = deepcopy(controller.state_manager)
        session["asked_priority_questions"] = set(
            controller.clarification_gen.asked_priority_questions
        )
        session["updated_at"] = datetime.now(timezone.utc)

    def start_diagnosis(self, query: str) -> Dict:
        with self._lock:
            now = datetime.now(timezone.utc)
            self._prune_expired(now)
            if len(self._sessions) >= self._max_sessions:
                raise SessionCapacityError("Maximum active diagnosis sessions reached")

            controller = self._get_controller()
            session_id = str(uuid4())
            session = {
                "state_manager": StateManager(),
                "asked_priority_questions": set(),
                "updated_at": now,
            }
            self._activate_session(controller, session)
            try:
                result = controller.start_diagnosis(query)
            except Exception:
                DIAGNOSIS_REQUESTS.labels(outcome="error").inc()
                raise
            self._save_session(session, controller)
            self._sessions[session_id] = session

            outcome = result.get("status", "error")
            DIAGNOSIS_REQUESTS.labels(outcome=outcome).inc()
            context = controller.state_manager.state.retrieved_context
            RETRIEVAL_OUTCOMES.labels(
                outcome="relevant" if context else "empty"
            ).inc()
            return {"session_id": session_id, **result}

    def submit_answer(self, session_id: str, answer: str) -> Dict:
        with self._lock:
            now = datetime.now(timezone.utc)
            self._prune_expired(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)

            controller = self._get_controller()
            self._activate_session(controller, session)
            if not controller.state_manager.state.initial_query:
                raise InvalidSessionStateError(
                    "Diagnosis has not been started for this session."
                )
            try:
                result = controller.submit_answer(answer)
            except Exception:
                DIAGNOSIS_REQUESTS.labels(outcome="error").inc()
                raise
            self._save_session(session, controller)
            self._sessions.move_to_end(session_id)

            outcome = result.get("status", "error")
            DIAGNOSIS_REQUESTS.labels(outcome=outcome).inc()
            context = controller.state_manager.state.retrieved_context
            RETRIEVAL_OUTCOMES.labels(
                outcome="relevant" if context else "empty"
            ).inc()
            return {"session_id": session_id, **result}

    def list_sessions(self) -> int:
        with self._lock:
            self._prune_expired(datetime.now(timezone.utc))
            return len(self._sessions)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._prune_expired(datetime.now(timezone.utc))
            if self._sessions.pop(session_id, None) is None:
                raise SessionNotFoundError(session_id)

    def search_disease_knowledge(self, query: str, limit: int = 5) -> list[Dict]:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20.")
        with self._lock:
            try:
                controller = self._get_controller()
                raw = controller.retriever.retrieve(query, [])
                filtered = controller.retrieval_evaluator.evaluate_chunks(
                    raw, query
                )
            except Exception as exc:
                logger.error(
                    "disease_knowledge_search_unavailable",
                    extra={"error_type": type(exc).__name__},
                )
                raise KnowledgeUnavailableError(
                    "Disease-document retrieval is currently unavailable."
                ) from exc
            return filtered[:limit]

class IncidentService:
    """SQLite-backed incident store with explicit acknowledgement and resolution."""

    def __init__(self, db_path=None):
        self._db_path = str(db_path or settings.INCIDENT_DB_PATH)
        self._connection = None
        self._lock = RLock()

    def _connect(self):
        if self._connection is None:
            if self._db_path != ":memory:":
                from pathlib import Path

                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self._db_path, check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolution TEXT
                )
                """
            )
            self._connection.commit()
        return self._connection

    @staticmethod
    def _from_row(row) -> IncidentRecord:
        return IncidentRecord(
            id=UUID(row["id"]),
            title=row["title"],
            severity=row["severity"],
            description=row["description"],
            status=row["status"],
            evidence=json.loads(row["evidence_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            resolution=row["resolution"],
        )

    def create(self, request: IncidentCreateRequest) -> IncidentRecord:
        now = datetime.now(timezone.utc)
        incident = IncidentRecord(
            id=uuid4(),
            title=request.title,
            severity=request.severity,
            description=request.description,
            status="open",
            evidence=request.evidence,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._connect().execute(
                """
                INSERT INTO incidents
                    (id, title, severity, description, status, evidence_json,
                     created_at, updated_at, resolution)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(incident.id),
                    incident.title,
                    incident.severity,
                    incident.description,
                    incident.status,
                    json.dumps(incident.evidence, sort_keys=True),
                    incident.created_at.isoformat(),
                    incident.updated_at.isoformat(),
                    incident.resolution,
                ),
            )
            self._connection.commit()
            self._refresh_metrics()
        return incident

    def list(self) -> list[IncidentRecord]:
        with self._lock:
            rows = self._connect().execute(
                "SELECT * FROM incidents ORDER BY created_at DESC LIMIT 1000"
            ).fetchall()
            return [self._from_row(row) for row in rows]

    def get(self, incident_id: UUID) -> IncidentRecord:
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM incidents WHERE id = ?", (str(incident_id),)
            ).fetchone()
            if row is None:
                raise IncidentNotFoundError(str(incident_id))
            return self._from_row(row)

    def find_active_signal(self, signal: str) -> Optional[IncidentRecord]:
        with self._lock:
            rows = self._connect().execute(
                "SELECT * FROM incidents WHERE status IN ('open', 'acknowledged') "
                "ORDER BY created_at DESC"
            ).fetchall()
            for row in rows:
                incident = self._from_row(row)
                if incident.evidence.get("signal") == signal:
                    return incident
            return None

    def add_triage(self, incident_id: UUID, triage: Dict) -> IncidentRecord:
        with self._lock:
            connection = self._connect()
            row = connection.execute(
                "SELECT * FROM incidents WHERE id = ?", (str(incident_id),)
            ).fetchone()
            if row is None:
                raise IncidentNotFoundError(str(incident_id))
            incident = self._from_row(row)
            evidence = {**incident.evidence, **triage}
            updated_at = datetime.now(timezone.utc)
            connection.execute(
                "UPDATE incidents SET evidence_json = ?, updated_at = ? WHERE id = ?",
                (
                    json.dumps(evidence, sort_keys=True),
                    updated_at.isoformat(),
                    str(incident_id),
                ),
            )
            connection.commit()
            updated = incident.model_copy(
                update={"evidence": evidence, "updated_at": updated_at}
            )
            return updated

    def acknowledge(self, incident_id: UUID) -> IncidentRecord:
        return self._update(incident_id, status="acknowledged")

    def resolve(self, incident_id: UUID, resolution: str) -> IncidentRecord:
        return self._update(
            incident_id, status="resolved", resolution=resolution
        )

    def _update(self, incident_id: UUID, **changes) -> IncidentRecord:
        with self._lock:
            connection = self._connect()
            existing = connection.execute(
                "SELECT * FROM incidents WHERE id = ?", (str(incident_id),)
            ).fetchone()
            if existing is None:
                raise IncidentNotFoundError(str(incident_id))
            incident = self._from_row(existing)
            if incident.status == "resolved":
                raise IncidentStateConflictError(
                    "Resolved incidents cannot be changed."
                )
            updated = incident.model_copy(
                update={**changes, "updated_at": datetime.now(timezone.utc)}
            )
            connection.execute(
                """
                UPDATE incidents
                SET status = ?, updated_at = ?, resolution = ?
                WHERE id = ?
                """,
                (
                    updated.status,
                    updated.updated_at.isoformat(),
                    updated.resolution,
                    str(incident_id),
                ),
            )
            connection.commit()
            self._refresh_metrics()
            return updated

    def _refresh_metrics(self) -> None:
        INCIDENTS.clear()
        for incident in self.list():
            INCIDENTS.labels(
                status=incident.status, severity=incident.severity
            ).inc()
