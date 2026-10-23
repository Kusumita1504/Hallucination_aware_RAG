# Knowledge or model backend unavailable

## Evidence to collect

- Check `/health/ready` for missing local disease data and inspect application initialization logs.
- Confirm the configured PDF directory contains the expected coffee references and `data/disease_knowledge.json` is readable.
- Check that FAISS index files are present or allow a first-run index build to complete.
- If generation or clarification failed, inspect `LLM_PROVIDER`, endpoint URL, model name, and configured request timeout without exposing API keys.

## Troubleshooting suggestions

- Treat missing retrieval evidence as a safe fallback condition; do not invent a diagnosis.
- Re-run the existing labeled evaluation after a knowledge/index change and review its data source and case count.
- Record evidence and obtain operator review before changing model or knowledge data.

## Safety

This runbook cannot authorize data modification, model deployment, or process control.
