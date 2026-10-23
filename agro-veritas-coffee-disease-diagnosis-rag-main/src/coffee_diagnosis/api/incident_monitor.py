"""Rolling-window SLI threshold detection and recent error telemetry."""

from collections import deque
from datetime import datetime, timezone
import math
from threading import RLock
import time
from typing import Deque, Dict, List, Optional

from config import settings
from coffee_diagnosis.api.schemas import IncidentCreateRequest
from coffee_diagnosis.api.service import IncidentService


class IncidentMonitor:
    """Detects measured error-rate and p95 latency threshold breaches."""

    def __init__(
        self,
        incident_service: IncidentService,
        error_rate_threshold: Optional[float] = None,
        latency_threshold_seconds: Optional[float] = None,
        window_seconds: Optional[int] = None,
        min_requests: Optional[int] = None,
    ):
        self.incident_service = incident_service
        self.error_rate_threshold = (
            settings.INCIDENT_ERROR_RATE_THRESHOLD
            if error_rate_threshold is None
            else error_rate_threshold
        )
        self.latency_threshold_seconds = (
            settings.INCIDENT_LATENCY_THRESHOLD_SECONDS
            if latency_threshold_seconds is None
            else latency_threshold_seconds
        )
        self.window_seconds = (
            settings.INCIDENT_WINDOW_SECONDS
            if window_seconds is None
            else window_seconds
        )
        self.min_requests = (
            settings.INCIDENT_MIN_REQUESTS if min_requests is None else min_requests
        )
        if not 0 < self.error_rate_threshold <= 1:
            raise ValueError("error_rate_threshold must be in (0, 1].")
        if self.latency_threshold_seconds <= 0:
            raise ValueError("latency_threshold_seconds must be greater than zero.")
        if self.window_seconds <= 0 or self.min_requests <= 0:
            raise ValueError("window_seconds and min_requests must be positive.")

        self._requests: Deque[Dict] = deque(maxlen=10000)
        self._errors: Deque[Dict] = deque(maxlen=200)
        self._lock = RLock()

    def observe(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
        request_id: str,
    ) -> List[Dict]:
        now = time.monotonic()
        event = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_seconds": round(duration_seconds, 6),
            "request_id": request_id,
        }
        with self._lock:
            self._requests.append(
                {
                    "monotonic": now,
                    "status_code": status_code,
                    "duration_seconds": duration_seconds,
                }
            )
            if status_code >= 500:
                self._errors.append({**event, "_monotonic": now})
            self._prune(now)
            return self._detect_breaches()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._requests and self._requests[0]["monotonic"] < cutoff:
            self._requests.popleft()
        while self._errors and self._errors[0]["_monotonic"] < cutoff:
            self._errors.popleft()

    def _detect_breaches(self) -> List[Dict]:
        requests = list(self._requests)
        if len(requests) < self.min_requests:
            return []

        errors = sum(1 for item in requests if item["status_code"] >= 500)
        error_rate = errors / len(requests)
        durations = sorted(item["duration_seconds"] for item in requests)
        p95_index = max(0, math.ceil(0.95 * len(durations)) - 1)
        p95_latency = durations[p95_index]

        signals = []
        if error_rate >= self.error_rate_threshold:
            signals.append(
                (
                    "error_rate",
                    "API 5xx error-rate threshold exceeded",
                    "critical" if error_rate >= min(1, self.error_rate_threshold * 2) else "high",
                    error_rate,
                    self.error_rate_threshold,
                )
            )
        if p95_latency >= self.latency_threshold_seconds:
            signals.append(
                (
                    "latency",
                    "API p95 latency threshold exceeded",
                    "high",
                    p95_latency,
                    self.latency_threshold_seconds,
                )
            )

        created = []
        for signal, title, severity, observed, threshold in signals:
            if self.incident_service.find_active_signal(signal):
                continue
            evidence = {
                "signal": signal,
                "source": "measured_request_telemetry",
                "window_seconds": self.window_seconds,
                "request_count": len(requests),
                "error_count": errors,
                "error_rate": round(error_rate, 6),
                "p95_latency_seconds": round(p95_latency, 6),
                "observed_value": round(observed, 6),
                "threshold": threshold,
                "recent_errors": [
                    {key: value for key, value in error.items() if key != "_monotonic"}
                    for error in list(self._errors)[-10:]
                ],
                "dashboard_url": settings.INCIDENT_DASHBOARD_URL,
                "simulated": False,
            }
            created.append(
                self.incident_service.create(
                    IncidentCreateRequest(
                        title=title,
                        severity=severity,
                        description=(
                            f"Measured {signal} value {observed:.4f} breached "
                            f"configured threshold {threshold:.4f} over "
                            f"{self.window_seconds} seconds."
                        ),
                        evidence=evidence,
                    )
                ).model_dump(mode="json")
            )
        return created

    def recent_errors(self, limit: int = 50) -> List[Dict]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        with self._lock:
            return [
                {key: value for key, value in error.items() if key != "_monotonic"}
                for error in list(self._errors)[-limit:][::-1]
            ]

    def telemetry_snapshot(self) -> Dict:
        with self._lock:
            self._prune(time.monotonic())
            requests = list(self._requests)
            errors = sum(1 for item in requests if item["status_code"] >= 500)
            durations = sorted(item["duration_seconds"] for item in requests)
            p95 = (
                durations[max(0, math.ceil(0.95 * len(durations)) - 1)]
                if durations
                else None
            )
            return {
                "source": "measured_request_telemetry",
                "simulated": False,
                "window_seconds": self.window_seconds,
                "request_count": len(requests),
                "error_count": errors,
                "error_rate": errors / len(requests) if requests else 0.0,
                "p95_latency_seconds": p95,
            }
