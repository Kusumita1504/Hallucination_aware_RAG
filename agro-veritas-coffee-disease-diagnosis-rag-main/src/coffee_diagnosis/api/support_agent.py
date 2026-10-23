"""Evidence-grounded support tools and incident-triage summaries."""

from concurrent.futures import ThreadPoolExecutor
import logging
import re
from typing import Dict, List

from config import settings
from anthropic import APIError as AnthropicAPIError
from coffee_diagnosis.api.incident_monitor import IncidentMonitor
from coffee_diagnosis.core.llm_client import LLMClient
from coffee_diagnosis.api.service import (
    DiagnosisService,
    IncidentService,
    KnowledgeUnavailableError,
)
from openai import APIError as OpenAIAPIError

logger = logging.getLogger("coffee_guard.support")

_DISEASE_QUERY_TERMS = {
    "coffee", "leaf", "leaves", "plant", "berry", "berries", "rust", "wilt",
    "spots", "yellow", "orange", "disease", "symptom", "symptoms", "root",
}


def _terms(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2
    }


class SupportAgent:
    """Combines operational signals, runbooks, disease evidence, and incident history."""

    def __init__(
        self,
        diagnosis_service: DiagnosisService,
        incident_service: IncidentService,
        incident_monitor: IncidentMonitor,
    ):
        self.diagnosis_service = diagnosis_service
        self.incident_service = incident_service
        self.incident_monitor = incident_monitor
        self.tools = {
            "service_health": self.service_health,
            "recent_errors": self.recent_errors,
            "knowledge_search": self.knowledge_search,
            "incident_history": self.incident_history,
        }

    def service_health(self) -> Dict:
        missing = []
        if not settings.PDF_DIR.is_dir() or not any(settings.PDF_DIR.glob("*.pdf")):
            missing.append("PDF knowledge files")
        if not (settings.DATA_DIR / "disease_knowledge.json").is_file():
            missing.append("structured disease knowledge")
        return {
            "status": "ready" if not missing else "not_ready",
            "missing": missing,
            "controller_initialized": (
                self.diagnosis_service._controller is not None
            ),
            **self.incident_monitor.telemetry_snapshot(),
        }

    def recent_errors(self, limit: int = 20) -> Dict:
        return {
            "source": "measured_request_telemetry",
            "simulated": False,
            "errors": self.incident_monitor.recent_errors(limit),
        }

    def incident_history(self, query: str, limit: int = 5) -> Dict:
        terms = _terms(query)
        scored = []
        for incident in self.incident_service.list():
            haystack = " ".join(
                [
                    incident.title,
                    incident.description,
                    incident.status,
                    str(incident.evidence),
                ]
            )
            overlap = len(terms & _terms(haystack))
            if overlap:
                scored.append((overlap, incident))
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return {
            "incidents": [
                {
                    "id": str(incident.id),
                    "title": incident.title,
                    "severity": incident.severity,
                    "status": incident.status,
                    "description": incident.description,
                    "evidence": incident.evidence,
                    "created_at": incident.created_at.isoformat(),
                    "relevance": score / max(len(terms), 1),
                    "source": f"incident:{incident.id}",
                }
                for score, incident in scored[:limit]
            ]
        }

    def knowledge_search(self, query: str, limit: int = 5) -> Dict:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20.")
        terms = _terms(query)
        results = []
        retrieval_error = None
        disease_retrieval_status = "not_applicable"

        runbook_dir = settings.RUNBOOK_DIR
        if runbook_dir.is_dir():
            for path in sorted(runbook_dir.glob("*.md")):
                content = path.read_text(encoding="utf-8")
                score = len(terms & _terms(f"{path.stem} {content}"))
                if score:
                    results.append(
                        {
                            "content": content,
                            "source": f"runbook:{path.name}",
                            "source_type": "RUNBOOK",
                            "relevance_score": score / max(len(terms), 1),
                        }
                    )

        if terms & _DISEASE_QUERY_TERMS:
            try:
                disease_chunks = self.diagnosis_service.search_disease_knowledge(
                    query, limit=limit
                )
            except KnowledgeUnavailableError:
                disease_chunks = []
                retrieval_error = "disease_document_retrieval_unavailable"
                disease_retrieval_status = "unavailable"
            else:
                disease_retrieval_status = "available"
        else:
            disease_chunks = []
        for chunk in disease_chunks:
            results.append(
                {
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "source_type": chunk["source_type"],
                    "metadata": chunk.get("metadata", {}),
                    "relevance_score": chunk.get("relevance_score", 0.0),
                }
            )

        results.sort(key=lambda result: result["relevance_score"], reverse=True)
        return {
            "query": query,
            "results": results[:limit],
            "disease_retrieval_error": retrieval_error,
            "disease_retrieval_status": disease_retrieval_status,
        }

    def search_incidents(self, query: str, limit: int = 5) -> Dict:
        return self.incident_history(query, limit)

    def invoke_tools(self, calls: List[Dict]) -> Dict:
        """Invoke independent read-only tools and return named, validated results."""
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(calls) or 1, 4)) as executor:
            pending = {
                name: executor.submit(self._invoke, name, arguments)
                for name, arguments in calls
            }
            for name, future in pending.items():
                results[name] = future.result()
        return results

    def _invoke(self, name: str, arguments: Dict):
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown support tool: {name}")
        if name == "service_health":
            return tool()
        if name == "recent_errors":
            return tool(**arguments)
        if name == "incident_history":
            return tool(**arguments)
        if name == "knowledge_search":
            return tool(**arguments)
        raise ValueError(f"Tool {name!r} is not available.")

    def investigate(self, query: str) -> Dict:
        tool_results = self.invoke_tools(
            [
                ("service_health", {}),
                ("recent_errors", {"limit": 10}),
                ("knowledge_search", {"query": query, "limit": 5}),
                ("incident_history", {"query": query, "limit": 5}),
            ]
        )
        evidence = []
        for item in tool_results["knowledge_search"]["results"]:
            evidence.append(
                {
                    "id": len(evidence) + 1,
                    "source": item["source"],
                    "source_type": item["source_type"],
                    "content": item["content"],
                    "relevance_score": item["relevance_score"],
                }
            )
        for incident in tool_results["incident_history"]["incidents"]:
            evidence.append(
                {
                    "id": len(evidence) + 1,
                    "source": incident["source"],
                    "source_type": "INCIDENT",
                    "content": incident["description"],
                    "relevance_score": incident["relevance"],
                }
            )

        summary = self._evidence_summary(query, tool_results, evidence)
        summary_mode = "evidence_template"
        llm_status = "disabled"
        if settings.SUPPORT_LLM_ENABLED:
            try:
                candidate = LLMClient().chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Summarize only the supplied operational evidence and runbook "
                                "guidance. Cite every factual claim using the provided [n] IDs. "
                                "Do not invent causes, claim actions were performed, or authorize "
                                "remediation. State uncertainty when evidence is insufficient."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {query}\n"
                                f"Tool results: {tool_results}\n"
                                f"Citable evidence: {evidence}"
                            ),
                        },
                    ],
                    max_tokens=400,
                )
            except (OpenAIAPIError, AnthropicAPIError, ValueError) as exc:
                logger.warning(
                    "support_summary_provider_unavailable",
                    extra={"error_type": type(exc).__name__},
                )
                llm_status = "provider_unavailable"
            else:
                cited_ids = {
                    int(value) for value in re.findall(r"\[(\d+)\]", candidate)
                }
                valid_ids = {item["id"] for item in evidence}
                if candidate.strip() and cited_ids and cited_ids.issubset(valid_ids):
                    summary = candidate.strip()
                    summary_mode = "llm_with_validated_citations"
                    llm_status = "used"
                else:
                    llm_status = "invalid_or_uncited_response"
        return {
            "query": query,
            "summary": summary,
            "tool_results": tool_results,
            "evidence": evidence,
            "summary_mode": summary_mode,
            "llm_status": llm_status,
        }

    @staticmethod
    def _evidence_summary(query: str, results: Dict, evidence: List[Dict]) -> str:
        health = results["service_health"]
        errors = results["recent_errors"]["errors"]
        incidents = results["incident_history"]["incidents"]
        status = health["status"]
        lines = [
            f"Investigation for: {query}",
            f"Service readiness: {status}.",
            (
                f"Recent measured 5xx events: {len(errors)}."
                if errors
                else "No recent measured 5xx events were recorded."
            ),
            (
                f"Relevant historical incidents: {len(incidents)}."
                if incidents
                else "No relevant historical incidents were found."
            ),
        ]
        if evidence:
            citations = ", ".join(
                f"[{item['id']}] {item['source']}" for item in evidence
            )
            lines.append(f"Retrieved evidence: {citations}.")
            lines.append(
                "Troubleshooting guidance is limited to the cited runbooks and incident evidence; "
                "confirm the measurements before taking action."
            )
        else:
            lines.append(
                "No sufficiently relevant documentation or incident evidence was found; "
                "no diagnosis or remediation is suggested."
            )
        lines.append("No operational remediation was executed or authorized.")
        return " ".join(lines)
