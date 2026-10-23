# API latency triage

## Scope

Use when measured p95 HTTP request latency exceeds the configured threshold.

## Evidence to collect

- Review the incident window, request volume, p95 latency, and threshold.
- Compare latency by route in Prometheus and identify whether diagnosis or non-diagnosis requests are affected.
- Check request IDs and status codes from recent errors; do not infer latency causes from diagnosis text.
- Check readiness, CPU/memory at the deployment layer, vector-index initialization, and configured LLM endpoint responsiveness.

## Troubleshooting suggestions

- Separate slow cold-start index loading from sustained request latency.
- Check the LLM client timeout and provider health when generation is the slow path.
- Compare before/after latency only across equivalent windows and traffic.
- Acknowledge the incident and document the measured recovery signal after operator action.

## Safety

Suggestions are advisory. Do not restart processes, resize resources, or alter indexes without explicit human approval.
