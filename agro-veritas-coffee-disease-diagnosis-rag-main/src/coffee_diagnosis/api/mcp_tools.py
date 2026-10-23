"""Validated tool implementations shared by the standalone MCP server."""

from typing import Any, Dict

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, UUID4, ValidationError

from coffee_diagnosis.api.incident_monitor import IncidentMonitor
from coffee_diagnosis.api.schemas import (
    AnswerRequest,
    DiagnosisRequest,
    InvestigationRequest,
    KnowledgeSearchRequest,
)
from coffee_diagnosis.api.service import (
    DiagnosisService,
    IncidentService,
    KnowledgeUnavailableError,
    SessionNotFoundError,
)
from coffee_diagnosis.api.support_agent import SupportAgent


class MCPToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    result: Dict[str, Any] | None = None
    error: str | None = None
    message: str | None = None


class MCPAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID4
    answer: str


class MCPTools:
    """Read-only support tools plus the existing session-based diagnosis flow."""

    def __init__(
        self,
        diagnosis_service: DiagnosisService,
        incident_service: IncidentService,
        incident_monitor: IncidentMonitor,
    ):
        self.diagnosis_service = diagnosis_service
        self.incident_service = incident_service
        self.incident_monitor = incident_monitor
        self.support_agent = SupportAgent(
            diagnosis_service, incident_service, incident_monitor
        )

    @staticmethod
    def _validate_output(value: Dict[str, Any]) -> Dict[str, Any]:
        return MCPToolResponse.model_validate(
            jsonable_encoder(value)
        ).model_dump(exclude_none=True)

    def diagnose(self, query: str) -> Dict[str, Any]:
        try:
            request = DiagnosisRequest(query=query)
        except ValidationError as exc:
            return self._validate_output(
                {"ok": False, "error": "invalid_input", "message": "Input failed validation."}
            )
        try:
            return self._validate_output(
                {"ok": True, "result": self.diagnosis_service.start_diagnosis(request.query)}
            )
        except (KnowledgeUnavailableError, RuntimeError):
            return self._validate_output(
                {
                    "ok": False,
                    "error": "diagnosis_unavailable",
                    "message": "The diagnosis service is temporarily unavailable.",
                }
            )

    def submit_diagnosis_answer(
        self, session_id: str, answer: str
    ) -> Dict[str, Any]:
        try:
            request = MCPAnswerRequest(
                session_id=session_id, answer=AnswerRequest(answer=answer).answer
            )
        except ValidationError as exc:
            return self._validate_output(
                {"ok": False, "error": "invalid_input", "message": "Input failed validation."}
            )
        try:
            result = self.diagnosis_service.submit_answer(
                str(request.session_id), request.answer
            )
        except SessionNotFoundError:
            return self._validate_output(
                {
                    "ok": False,
                    "error": "session_not_found",
                    "message": "Diagnosis session does not exist or has expired.",
                }
            )
        return self._validate_output({"ok": True, "result": result})

    def search_knowledge(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            request = KnowledgeSearchRequest(query=query, limit=limit)
        except ValidationError as exc:
            return self._validate_output(
                {"ok": False, "error": "invalid_input", "message": "Input failed validation."}
            )
        return self._validate_output(
            {
                "ok": True,
                "result": self.support_agent.knowledge_search(
                    request.query, request.limit
                ),
            }
        )

    def service_health(self) -> Dict[str, Any]:
        return self._validate_output(
            {"ok": True, "result": self.support_agent.service_health()}
        )

    def recent_errors(self, limit: int = 20) -> Dict[str, Any]:
        if not 1 <= limit <= 200:
            return self._validate_output(
                {
                    "ok": False,
                    "error": "invalid_input",
                    "message": "limit must be between 1 and 200.",
                }
            )
        return self._validate_output(
            {"ok": True, "result": self.support_agent.recent_errors(limit)}
        )

    def incident_history(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            request = InvestigationRequest(query=query)
        except ValidationError as exc:
            return self._validate_output(
                {"ok": False, "error": "invalid_input", "message": "Input failed validation."}
            )
        if not 1 <= limit <= 20:
            return self._validate_output(
                {
                    "ok": False,
                    "error": "invalid_input",
                    "message": "limit must be between 1 and 20.",
                }
            )
        return self._validate_output(
            {
                "ok": True,
                "result": self.support_agent.incident_history(
                    request.query, limit
                ),
            }
        )

    def investigate(self, query: str) -> Dict[str, Any]:
        try:
            request = InvestigationRequest(query=query)
        except ValidationError as exc:
            return self._validate_output(
                {"ok": False, "error": "invalid_input", "message": "Input failed validation."}
            )
        return self._validate_output(
            {"ok": True, "result": self.support_agent.investigate(request.query)}
        )
