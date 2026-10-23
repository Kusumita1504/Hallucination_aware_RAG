"""Fast, dependency-light tests for the HTTP API and its session isolation."""

import sys
import json
import hashlib
import hmac
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from coffee_diagnosis.api.app import create_app
from coffee_diagnosis.api.service import DiagnosisService, IncidentService
from coffee_diagnosis.api.incident_monitor import IncidentMonitor
from coffee_diagnosis.api.schemas import IncidentCreateRequest
from coffee_diagnosis.api.mcp_tools import MCPTools
from coffee_diagnosis.rag.retrieval_evaluator import RetrievalEvaluator
from coffee_diagnosis.rag.state_manager import StateManager


class FakeController:
    def __init__(self):
        self.state_manager = StateManager()
        self.clarification_gen = SimpleNamespace(asked_priority_questions=set())
        self.retriever = SimpleNamespace(
            retrieve=lambda query, previous_answers: [
                {
                    "content": "Coffee leaf rust produces orange powder on the underside of leaves.",
                    "source": "disease_knowledge.json",
                    "source_type": "JSON",
                    "metadata": {"disease_name": "Coffee Leaf Rust"},
                }
            ]
        )
        self.retrieval_evaluator = RetrievalEvaluator(relevance_threshold=0)

    def start_diagnosis(self, query):
        self.state_manager.initialize(query)
        self.clarification_gen.asked_priority_questions.clear()
        self.state_manager.add_system_question("Where are the symptoms?")
        return {"status": "question", "question": "Where are the symptoms?"}

    def submit_answer(self, answer):
        self.state_manager.add_user_response(answer)
        return {
            "status": "diagnosis",
            "diagnosis": {
                "disease_name": "Example",
                "evidence": [self.state_manager.state.initial_query, answer],
            },
        }


def make_client():
    service = DiagnosisService(controller_factory=FakeController)
    return TestClient(
        create_app(
            diagnosis_service=service,
            incident_service=IncidentService(":memory:"),
        )
    )


def test_health_and_metrics_endpoints():
    client = make_client()
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/metrics").status_code == 200
    assert "coffee_guard_http_requests_total" in client.get("/metrics").text


def test_diagnosis_validation_rejects_blank_and_extra_fields():
    client = make_client()
    blank = client.post("/api/v1/diagnosis", json={"query": "  "})
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "validation_error"
    extra = client.post(
        "/api/v1/diagnosis", json={"query": "orange spots", "other": "value"}
    )
    assert extra.status_code == 422


def test_multi_turn_sessions_keep_conversations_isolated():
    client = make_client()
    first = client.post("/api/v1/diagnosis", json={"query": "first symptoms"})
    second = client.post("/api/v1/diagnosis", json={"query": "second symptoms"})
    first_id = first.json()["session_id"]
    second_id = second.json()["session_id"]

    first_result = client.post(
        f"/api/v1/diagnosis/{first_id}/answers", json={"answer": "first answer"}
    )
    second_result = client.post(
        f"/api/v1/diagnosis/{second_id}/answers", json={"answer": "second answer"}
    )

    assert first_result.json()["diagnosis"]["evidence"] == [
        "first symptoms",
        "first answer",
    ]
    assert second_result.json()["diagnosis"]["evidence"] == [
        "second symptoms",
        "second answer",
    ]


def test_unknown_and_invalid_session_ids_have_consistent_errors():
    client = make_client()
    invalid = client.post("/api/v1/diagnosis/not-a-uuid/answers", json={"answer": "yes"})
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_session_id"

    unknown = client.post(
        "/api/v1/diagnosis/00000000-0000-0000-0000-000000000001/answers",
        json={"answer": "yes"},
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "session_not_found"


def test_incident_create_acknowledge_and_resolve():
    client = make_client()
    created = client.post(
        "/api/v1/incidents",
        json={
            "title": "High API latency",
            "severity": "high",
            "description": "Latency exceeded the configured threshold.",
            "evidence": {"source": "test", "simulated": True},
        },
    )
    assert created.status_code == 201
    incident_id = created.json()["id"]
    assert created.json()["status"] == "open"
    assert created.json()["evidence"]["triage_status"] == "complete"
    assert any(
        source.startswith("runbook:")
        for source in created.json()["evidence"]["triage_sources"]
    )

    acknowledged = client.post(
        f"/api/v1/incidents/{incident_id}/acknowledge"
    )
    assert acknowledged.json()["status"] == "acknowledged"
    resolved = client.post(
        f"/api/v1/incidents/{incident_id}/resolve",
        json={"resolution": "Threshold returned to normal."},
    )
    assert resolved.json()["status"] == "resolved"
    assert resolved.json()["resolution"] == "Threshold returned to normal."
    assert len(client.get("/api/v1/incidents").json()) == 1
    conflict = client.post(
        f"/api/v1/incidents/{incident_id}/acknowledge"
    )
    assert conflict.status_code == 409


def test_incidents_persist_between_service_instances(tmp_path):
    database_path = tmp_path / "incidents.sqlite3"
    first_service = IncidentService(database_path)
    incident = first_service.create(
        IncidentCreateRequest(
            title="Persistent incident",
            severity="medium",
            description="Persist this incident record.",
            evidence={"signal": "test"},
        )
    )
    second_service = IncidentService(database_path)
    assert second_service.get(incident.id).title == "Persistent incident"
    assert second_service.acknowledge(incident.id).status == "acknowledged"
    assert IncidentService(database_path).get(incident.id).status == "acknowledged"


def test_threshold_monitor_detects_and_deduplicates_measured_errors():
    incidents = IncidentService(":memory:")
    monitor = IncidentMonitor(
        incidents,
        error_rate_threshold=0.5,
        latency_threshold_seconds=10,
        window_seconds=60,
        min_requests=3,
    )
    assert monitor.observe("GET", "/health", 200, 0.01, "r1") == []
    assert monitor.observe("GET", "/api", 500, 0.02, "r2") == []
    created = monitor.observe("GET", "/api", 500, 0.03, "r3")

    assert len(created) == 1
    assert created[0]["evidence"]["source"] == "measured_request_telemetry"
    assert created[0]["evidence"]["simulated"] is False
    assert created[0]["evidence"]["error_rate"] == round(2 / 3, 6)
    assert monitor.observe("GET", "/api", 500, 0.04, "r4") == []
    assert len(incidents.list()) == 1
    assert len(monitor.recent_errors(10)) == 3


def test_latency_monitor_creates_incident_with_p95_evidence():
    incidents = IncidentService(":memory:")
    monitor = IncidentMonitor(
        incidents,
        error_rate_threshold=1,
        latency_threshold_seconds=0.5,
        window_seconds=60,
        min_requests=3,
    )
    monitor.observe("GET", "/health", 200, 0.01, "r1")
    monitor.observe("GET", "/health", 200, 0.02, "r2")
    created = monitor.observe("GET", "/diagnosis", 200, 0.8, "r3")

    assert len(created) == 1
    assert created[0]["evidence"]["signal"] == "latency"
    assert created[0]["evidence"]["p95_latency_seconds"] == 0.8
    assert monitor.telemetry_snapshot()["simulated"] is False


def test_recent_errors_endpoint_exposes_only_operational_metadata():
    client = make_client()
    client.get("/not-found")
    telemetry = client.get("/api/v1/telemetry").json()
    assert telemetry["source"] == "measured_request_telemetry"
    assert telemetry["simulated"] is False
    errors = client.get("/api/v1/errors/recent").json()
    assert errors["simulated"] is False
    assert errors["errors"] == []


def test_support_search_combines_runbook_and_disease_sources_with_citations():
    client = make_client()
    response = client.post(
        "/api/v1/knowledge/search",
        json={"query": "API error rate coffee leaf rust"},
    )
    assert response.status_code == 200
    sources = {item["source_type"] for item in response.json()["results"]}
    assert "RUNBOOK" in sources
    assert "JSON" in sources
    assert response.json()["disease_retrieval_error"] is None


def test_investigation_calls_multiple_tools_and_returns_evidence_backed_summary():
    client = make_client()
    client.post(
        "/api/v1/incidents",
        json={
            "title": "API 5xx error-rate threshold exceeded",
            "severity": "high",
            "description": "Measured API error rate exceeded the configured threshold.",
            "evidence": {"signal": "error_rate"},
        },
    )
    result = client.post(
        "/api/v1/support/investigate",
        json={"query": "API error rate"},
    )
    assert result.status_code == 200
    body = result.json()
    assert set(body["tool_results"]) == {
        "service_health",
        "recent_errors",
        "knowledge_search",
        "incident_history",
    }
    assert body["tool_results"]["incident_history"]["incidents"]
    assert "[1]" in body["summary"]
    assert body["evidence"][0]["source"]
    assert "No operational remediation was executed" in body["summary"]


def test_mcp_tools_validate_inputs_and_return_structured_results():
    incidents = IncidentService(":memory:")
    diagnosis = DiagnosisService(controller_factory=FakeController)
    tools = MCPTools(diagnosis, incidents, IncidentMonitor(incidents, min_requests=2))

    invalid = tools.search_knowledge("  ")
    assert invalid["ok"] is False
    assert invalid["error"] == "invalid_input"

    health = tools.service_health()
    assert health["ok"] is True
    assert health["result"]["simulated"] is False
    diagnosis_result = tools.diagnose("coffee leaves with orange powder")
    assert diagnosis_result["ok"] is True
    assert diagnosis_result["result"]["status"] == "question"
    answer = tools.submit_diagnosis_answer(
        diagnosis_result["result"]["session_id"], "underside"
    )
    assert answer["ok"] is True
    assert answer["result"]["status"] == "diagnosis"


def test_mcp_tools_report_unavailable_diagnosis_backend():
    incidents = IncidentService(":memory:")

    def unavailable_controller():
        raise RuntimeError("not available")

    tools = MCPTools(
        DiagnosisService(controller_factory=unavailable_controller),
        incidents,
        IncidentMonitor(incidents, min_requests=2),
    )
    result = tools.diagnose("coffee leaves with orange powder")
    assert result == {
        "ok": False,
        "error": "diagnosis_unavailable",
        "message": "The diagnosis service is temporarily unavailable.",
    }


def _slack_signature(secret, timestamp, body):
    base = f"v0:{timestamp}:".encode("utf-8") + body
    digest = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return f"v0={digest}"


def test_slack_commands_verify_signature_and_reject_replays(tmp_path):
    client = make_client()
    integration = client.app.state.slack
    integration.signing_secret = "test-secret"
    integration.event_db_path = tmp_path / "slack.sqlite3"
    timestamp = str(int(__import__("time").time()))
    body = urlencode(
        {
            "command": "/coffee",
            "text": "health",
            "trigger_id": "trigger-unique-1",
        }
    ).encode("utf-8")
    signature = _slack_signature("test-secret", timestamp, body)

    response = client.post(
        "/api/v1/slack/commands",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
    )
    assert response.status_code == 200
    assert "Service status" in response.json()["text"]

    replay = client.post(
        "/api/v1/slack/commands",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
    )
    assert replay.json() == response.json()

    invalid = client.post(
        "/api/v1/slack/commands",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": "v0=invalid",
        },
    )
    assert invalid.status_code == 401
    assert invalid.json()["error"]["code"] == "invalid_slack_request"


def test_slack_is_disabled_without_secret_and_runbook_command_works(tmp_path):
    client = make_client()
    integration = client.app.state.slack
    integration.signing_secret = ""
    disabled = client.post("/api/v1/slack/commands", data={"command": "/coffee"})
    assert disabled.status_code == 503

    integration.signing_secret = "test-secret"
    integration.event_db_path = tmp_path / "slack.sqlite3"
    timestamp = str(int(__import__("time").time()))
    body = urlencode(
        {
            "command": "/coffee",
            "text": "runbook API latency",
            "trigger_id": "trigger-runbook",
        }
    ).encode("utf-8")
    response = client.post(
        "/api/v1/slack/commands",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": _slack_signature(
                "test-secret", timestamp, body
            ),
        },
    )
    assert response.status_code == 200
    assert "api-latency.md" in response.json()["text"]


def test_request_size_limit_returns_structured_error(monkeypatch):
    import config.settings as project_settings

    monkeypatch.setattr(project_settings, "API_MAX_REQUEST_BYTES", 2)
    client = make_client()
    response = client.post(
        "/api/v1/diagnosis",
        json={"query": "too large"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_quality_metrics_require_explicit_labeled_evaluation(tmp_path, monkeypatch):
    import config.settings as project_settings

    client = make_client()
    unavailable = client.get("/api/v1/quality/evaluation")
    assert unavailable.status_code == 404
    assert "not used for quality metrics" in unavailable.json()["message"]

    results_path = tmp_path / "evaluation.json"
    results_path.write_text(
        json.dumps(
            {
                "data_source": "labeled_evaluation",
                "dataset": "held-out.json",
                "total_cases": 12,
                "evaluated_at": "2026-10-23T00:00:00+00:00",
                "evaluated_at_unix": 1792702800,
                "top1_multi_accuracy": 0.5,
                "top3_multi_accuracy": 0.75,
                "mrr": 0.6,
                "gt_hallucination_rate": 0.5,
                "semantic_precision_at_5": 0.4,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        project_settings, "DIAGNOSIS_EVALUATION_FILE", str(results_path)
    )
    evaluated_client = TestClient(create_app(diagnosis_service=DiagnosisService(FakeController)))
    response = evaluated_client.get("/api/v1/quality/evaluation")
    assert response.status_code == 200
    assert response.json()["data_source"] == "labeled_evaluation"
    assert response.json()["metrics"]["top1_multi_accuracy"] == 0.5
    metrics_text = evaluated_client.get("/metrics").text
    assert (
        'coffee_guard_labeled_top1_multi_accuracy{dataset="held-out.json"} 0.5'
        in metrics_text
    )
