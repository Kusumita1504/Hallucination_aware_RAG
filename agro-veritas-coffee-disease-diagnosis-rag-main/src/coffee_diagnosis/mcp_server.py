"""Standalone stdio MCP server exposing CoffeeGuard's diagnosis/support tools."""

from mcp.server.fastmcp import FastMCP

from coffee_diagnosis.api.incident_monitor import IncidentMonitor
from coffee_diagnosis.api.mcp_tools import MCPTools
from coffee_diagnosis.api.service import DiagnosisService, IncidentService

mcp = FastMCP("CoffeeGuard")
_incidents = IncidentService()
_diagnosis = DiagnosisService()
_monitor = IncidentMonitor(_incidents)
_tools = MCPTools(_diagnosis, _incidents, _monitor)


@mcp.tool()
def diagnose(query: str) -> dict:
    """Start a coffee-plant diagnosis; returns a question or evidence-grounded result."""
    return _tools.diagnose(query)


@mcp.tool()
def submit_diagnosis_answer(session_id: str, answer: str) -> dict:
    """Submit an answer to a multi-turn diagnosis session."""
    return _tools.submit_diagnosis_answer(session_id, answer)


@mcp.tool()
def knowledge_search(query: str, limit: int = 5) -> dict:
    """Search coffee disease references, operational runbooks, and incident reports."""
    return _tools.search_knowledge(query, limit)


@mcp.tool()
def service_health() -> dict:
    """Return local service readiness and measured operational telemetry."""
    return _tools.service_health()


@mcp.tool()
def recent_errors(limit: int = 20) -> dict:
    """Return recent measured HTTP 5xx metadata without request bodies."""
    return _tools.recent_errors(limit)


@mcp.tool()
def incident_history(query: str, limit: int = 5) -> dict:
    """Search historical incidents by query and return status and evidence."""
    return _tools.incident_history(query, limit)


@mcp.tool()
def investigate(query: str) -> dict:
    """Combine health, errors, knowledge search, and incident history into a cited summary."""
    return _tools.investigate(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
