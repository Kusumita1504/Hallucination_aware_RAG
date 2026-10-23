"""Signed Slack slash-command handling and optional incident notifications."""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs

import httpx

from config import settings
from coffee_diagnosis.api.support_agent import SupportAgent
from coffee_diagnosis.api.service import DiagnosisService

logger = logging.getLogger("coffee_guard.slack")


class SlackRequestError(ValueError):
    """Raised when a Slack request fails signature or command validation."""


class SlackNotificationError(RuntimeError):
    """Raised when configured Slack incident notification delivery fails."""


class SlackIntegration:
    def __init__(
        self,
        support_agent: SupportAgent,
        diagnosis_service: DiagnosisService,
        event_db_path: Optional[Path] = None,
        signing_secret: Optional[str] = None,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None,
    ):
        self.support_agent = support_agent
        self.diagnosis_service = diagnosis_service
        self.event_db_path = Path(event_db_path or settings.SLACK_EVENT_DB_PATH)
        self.signing_secret = (
            settings.SLACK_SIGNING_SECRET if signing_secret is None else signing_secret
        )
        self.bot_token = settings.SLACK_BOT_TOKEN if bot_token is None else bot_token
        self.channel_id = settings.SLACK_CHANNEL_ID if channel_id is None else channel_id

    def _connect(self):
        self.event_db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.event_db_path, timeout=5)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS slack_events (
                event_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                response_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        expiration = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        connection.execute(
            "DELETE FROM slack_events WHERE created_at < ?", (expiration,)
        )
        connection.commit()
        return connection

    def _verify_signature(
        self, raw_body: bytes, timestamp: str, signature: str
    ) -> None:
        if not self.signing_secret:
            raise SlackRequestError("Slack integration is not configured.")
        try:
            timestamp_value = int(timestamp)
        except (TypeError, ValueError) as exc:
            raise SlackRequestError("Invalid Slack request timestamp.") from exc
        now = int(time.time())
        if abs(now - timestamp_value) > settings.SLACK_TIMESTAMP_TOLERANCE_SECONDS:
            raise SlackRequestError("Slack request timestamp is outside the allowed window.")
        base = b"v0:" + timestamp.encode("ascii") + b":" + raw_body
        expected = "v0=" + hmac.new(
            self.signing_secret.encode("utf-8"), base, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature or ""):
            raise SlackRequestError("Slack request signature is invalid.")

    def _claim_event(self, event_id: str) -> Optional[Dict]:
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO slack_events
                    (event_id, status, response_json, created_at)
                VALUES (?, 'processing', NULL, ?)
                """,
                (event_id, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()
            if cursor.rowcount == 1:
                return None
            row = connection.execute(
                "SELECT status, response_json FROM slack_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row and row[0] == "complete" and row[1]:
                return json.loads(row[1])
        finally:
            connection.close()
        return {
            "response_type": "ephemeral",
            "text": "This command is already being processed.",
        }

    def _complete_event(self, event_id: str, response: Dict) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "UPDATE slack_events SET status = 'complete', response_json = ? "
                "WHERE event_id = ?",
                (json.dumps(response, sort_keys=True), event_id),
            )
            connection.commit()
        finally:
            connection.close()

    def _release_event(self, event_id: str) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "DELETE FROM slack_events WHERE event_id = ? AND status = 'processing'",
                (event_id,),
            )
            connection.commit()
        finally:
            connection.close()

    def handle_command(
        self,
        raw_body: bytes,
        timestamp: str,
        signature: str,
    ) -> Dict:
        self._verify_signature(raw_body, timestamp, signature)
        try:
            form = parse_qs(raw_body.decode("utf-8"), strict_parsing=True)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SlackRequestError("Slack command body is invalid.") from exc

        command = form.get("command", [""])[0]
        text = form.get("text", [""])[0].strip()
        trigger_id = form.get("trigger_id", [""])[0]
        if not command:
            raise SlackRequestError("Slack command is missing.")
        event_id = trigger_id or hashlib.sha256(raw_body).hexdigest()
        cached = self._claim_event(event_id)
        if cached is not None:
            return cached

        try:
            response = self._dispatch(command, text)
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "slack_command_failed",
                extra={"error_type": type(exc).__name__, "command": command},
            )
            response = {
                "response_type": "ephemeral",
                "text": "CoffeeGuard could not complete that command. Check service health and retry.",
            }
        except Exception:
            self._release_event(event_id)
            raise
        self._complete_event(event_id, response)
        return response

    def _dispatch(self, command: str, text: str) -> Dict:
        if command != "/coffee":
            raise SlackRequestError("Use the configured /coffee slash command.")
        parts = text.split(maxsplit=1)
        action = parts[0].lower() if parts else "help"
        argument = parts[1].strip() if len(parts) > 1 else ""
        if action == "health":
            result = self.support_agent.service_health()
            return self._reply(
                f"Service status: {result['status']}; "
                f"requests in window: {result['request_count']}; "
                f"5xx error rate: {result['error_rate']:.1%}; "
                f"p95 latency: {result['p95_latency_seconds']}s."
            )
        if action == "diagnose":
            if not argument:
                return self._reply("Usage: /coffee diagnose <coffee plant symptoms>")
            result = self.diagnosis_service.start_diagnosis(argument)
            if result.get("status") == "question":
                message = (
                    f"Clarification: {result['question']} "
                    f"(session `{result['session_id']}`)"
                )
            elif result.get("status") == "diagnosis":
                diagnosis = result["diagnosis"]
                message = (
                    f"Diagnosis: {diagnosis.disease_name}; "
                    f"confidence: {diagnosis.confidence:.0%}. "
                    f"Consult a local agronomist before treatment."
                )
            else:
                message = result.get(
                    "message", "The diagnosis service did not return a diagnosis."
                )
            return self._reply(message)
        if action == "runbook":
            if not argument:
                return self._reply("Usage: /coffee runbook <issue or symptom>")
            result = self.support_agent.knowledge_search(argument, limit=5)
            runbooks = [
                item for item in result["results"]
                if item["source_type"] == "RUNBOOK"
            ]
            if not runbooks:
                return self._reply("No relevant runbook was found.")
            return self._reply(
                "\n".join(
                    f"{item['source']}: {item['content'][:700]}"
                    for item in runbooks[:3]
                )
            )
        if action in {"incident", "incidents"}:
            incidents = self.support_agent.incident_history(
                argument or "open incidents", limit=5
            )["incidents"]
            if not incidents:
                return self._reply("No matching incidents were found.")
            return self._reply(
                "\n".join(
                    f"[{item['severity'].upper()}] {item['title']} — "
                    f"{item['status']} — {item['source']}"
                    for item in incidents
                )
            )
        return self._reply(
            "Commands: `/coffee diagnose <symptoms>`, `/coffee health`, "
            "`/coffee runbook <query>`, `/coffee incidents [query]`."
        )

    @staticmethod
    def _reply(text: str) -> Dict:
        return {"response_type": "ephemeral", "text": text[:3000]}

    async def notify_incident(self, incident: Dict) -> str:
        if not self.bot_token or not self.channel_id:
            return "disabled"
        evidence = incident.get("evidence", {})
        dashboard = evidence.get("dashboard_url", settings.INCIDENT_DASHBOARD_URL)
        text = (
            f"[{incident['severity'].upper()}] {incident['title']}\n"
            f"Status: {incident['status']}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=True)[:1800]}\n"
            f"Dashboard: {dashboard}"
        )
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                    json={"channel": self.channel_id, "text": text},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SlackNotificationError(
                f"Slack incident notification failed ({type(exc).__name__})."
            ) from exc
        if not payload.get("ok"):
            raise SlackNotificationError(
                "Slack rejected the incident notification."
            )
        return "sent"
