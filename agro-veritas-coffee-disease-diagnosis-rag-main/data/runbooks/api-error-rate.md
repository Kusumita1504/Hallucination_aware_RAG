# API 5xx error-rate triage

## Scope

Use this runbook when measured HTTP 5xx responses cross the configured error-rate threshold. This is a diagnosis guide; it does not authorize remediation.

## Evidence to collect

- Confirm the incident window, request count, 5xx count, observed error rate, and configured threshold in incident evidence.
- Use `/api/v1/errors/recent` to inspect timestamps, route templates, status codes, and request IDs. Request and symptom bodies are intentionally not logged.
- Correlate request IDs with the service logs and inspect the exception category without copying credentials or personal data.
- Check `/health/ready`, the Prometheus scrape target, and whether the issue is isolated to diagnosis requests or affects health and incident endpoints too.

## Troubleshooting suggestions

- If the diagnosis engine failed to initialize, check PDF/JSON data paths and vector-index permissions.
- If failures occur during generation, verify the configured local or optional hosted LLM endpoint and its availability.
- If errors are confined to malformed client input, review the validation response and client request schema.
- Acknowledge the incident, attach verified evidence, and document the human-approved resolution.

## Safety

Do not delete indexes, restart services, change data, or run remediation from an AI suggestion. Obtain explicit operator approval and follow the deployment's change process.
