"""Prometheus metrics for live service signals and offline labeled evaluation."""

import json
from pathlib import Path
from typing import Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()
HTTP_REQUESTS = Counter(
    "coffee_guard_http_requests_total",
    "Completed HTTP requests.",
    ("method", "path", "status"),
    registry=REGISTRY,
)
HTTP_ERRORS = Counter(
    "coffee_guard_http_errors_total",
    "HTTP requests returning an error status.",
    ("method", "path", "status"),
    registry=REGISTRY,
)
HTTP_REQUEST_DURATION = Histogram(
    "coffee_guard_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "path"),
    registry=REGISTRY,
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
DIAGNOSIS_REQUESTS = Counter(
    "coffee_guard_diagnosis_requests_total",
    "Diagnosis requests by outcome.",
    ("outcome",),
    registry=REGISTRY,
)
RETRIEVAL_OUTCOMES = Counter(
    "coffee_guard_retrieval_outcomes_total",
    "Diagnosis retrieval outcomes, based on evaluated context.",
    ("outcome",),
    registry=REGISTRY,
)
INCIDENTS = Gauge(
    "coffee_guard_incidents",
    "Current incidents by workflow status and severity.",
    ("status", "severity"),
    registry=REGISTRY,
)
EVALUATION_METRICS = {
    metric: Gauge(
        f"coffee_guard_labeled_{metric}",
        f"{metric.replace('_', ' ').capitalize()} from the labeled evaluation dataset.",
        ("dataset",),
        registry=REGISTRY,
    )
    for metric in (
        "top1_multi_accuracy",
        "top3_multi_accuracy",
        "mrr",
        "gt_hallucination_rate",
        "semantic_precision_at_5",
    )
}
EVALUATION_INFO = Gauge(
    "coffee_guard_labeled_evaluation_info",
    "Metadata for the loaded labeled evaluation result.",
    ("dataset", "cases", "data_source"),
    registry=REGISTRY,
)
EVALUATION_TIMESTAMP = Gauge(
    "coffee_guard_labeled_evaluation_timestamp_seconds",
    "Unix time when the labeled evaluation was run.",
    ("dataset",),
    registry=REGISTRY,
)

_evaluation_metadata: Optional[Dict] = None


def load_labeled_evaluation(path: Optional[str]) -> Optional[Dict]:
    """Load explicit offline evaluation output; live predictions are never aggregated."""
    global _evaluation_metadata
    if not path:
        _evaluation_metadata = None
        for gauge in EVALUATION_METRICS.values():
            gauge.clear()
        EVALUATION_INFO.clear()
        EVALUATION_TIMESTAMP.clear()
        return None

    evaluation_path = Path(path)
    try:
        with evaluation_path.open("r", encoding="utf-8") as result_file:
            results = json.load(result_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not load configured labeled evaluation file {evaluation_path}: {exc}"
        ) from exc

    if results.get("data_source") != "labeled_evaluation":
        raise ValueError(
            "Configured evaluation metrics must declare data_source='labeled_evaluation'."
        )
    if not isinstance(results.get("total_cases"), int) or results["total_cases"] < 1:
        raise ValueError("Labeled evaluation results must include total_cases >= 1.")

    metrics = results.get("metrics", results)
    dataset = str(results.get("dataset", evaluation_path.name))
    for name, gauge in EVALUATION_METRICS.items():
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Labeled evaluation metric {name!r} must be in [0, 1].")
        gauge.clear()
        gauge.labels(dataset=dataset).set(value)

    EVALUATION_INFO.clear()
    EVALUATION_INFO.labels(
        dataset=dataset,
        cases=str(results["total_cases"]),
        data_source="labeled_evaluation",
    ).set(1)
    EVALUATION_TIMESTAMP.clear()
    EVALUATION_TIMESTAMP.labels(dataset=dataset).set(
        float(results.get("evaluated_at_unix", 0))
    )
    _evaluation_metadata = {
        "data_source": "labeled_evaluation",
        "dataset": dataset,
        "total_cases": results["total_cases"],
        "evaluated_at": results.get("evaluated_at"),
        "metrics": {name: metrics[name] for name in EVALUATION_METRICS},
    }
    return dict(_evaluation_metadata)


def get_labeled_evaluation() -> Optional[Dict]:
    return dict(_evaluation_metadata) if _evaluation_metadata else None


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)
