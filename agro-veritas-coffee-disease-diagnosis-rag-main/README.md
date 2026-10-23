# Coffee Disease Diagnosis with Balanced Hybrid RAG

## Abstract

This repository contains a coffee disease diagnosis system built on a balanced hybrid Retrieval-Augmented Generation (RAG) architecture. The system combines structured JSON disease knowledge and unstructured PDF agronomy references, performs multi-turn clarification to resolve ambiguous symptom descriptions, and verifies generated diagnoses for consistency. The implementation is optimized for local inference by default through an OpenAI-compatible Ollama endpoint, with optional cloud fallback.

## 1. Objectives and Scope

### 1.1 Objectives

The project addresses three practical failure modes in agricultural diagnosis assistants:
1. single-disease retrieval bias,
2. premature diagnosis on incomplete symptom input,
3. unsupported or inconsistent model outputs.

### 1.2 Scope

The system provides decision support for coffee disease diagnosis from text-based symptom descriptions. It is intended for research and educational use, not as a replacement for expert agronomists or field diagnostics.

## 2. System Design

### 2.1 End-to-End Pipeline

The diagnosis flow is:
1. ambiguity detection from user query,
2. multi-query expansion,
3. hybrid retrieval from JSON and PDF sources,
4. CRAG-style relevance filtering,
5. clarification question generation,
6. multi-turn interaction loop,
7. structured diagnosis generation,
8. consistency/hallucination verification.

### 2.2 Clarification Behavior

The multi-turn interface accepts:
- detailed symptom responses,
- short binary responses (e.g., `yes`, `no`, `partly`),
- uncertainty responses (e.g., `not sure`, `don't know`).

This allows natural user interaction without forcing long symptom restatements on every turn.

## 3. Dataset and Retrieval Strategy

### 3.1 Data Sources

- Structured dataset: `data/disease_knowledge.json` (15 diseases).
- Document corpus: PDFs under `data/pdfs/`.

### 3.2 Hybrid Retrieval Ratio

- 60% JSON retrieval (balanced disease coverage),
- 40% PDF retrieval (supporting evidence and detail).

### 3.3 Transparency

Evidence is tagged by source type (`[JSON]` and `[PDF]`) in the UI and internal context payloads.

## 4. Core Implementation Modules

### 4.1 `src/coffee_diagnosis/core`

- `pdf_loader.py`: PDF loading and chunking.
- `vector_store.py`: FAISS index build/load/search.
- `json_retriever.py`: structured disease retrieval.
- `retriever.py`: hybrid merge/rank logic.
- `llm_client.py`: LLM provider abstraction.

### 4.2 `src/coffee_diagnosis/rag`

- `ambiguity_detector.py`: missing-attribute detection.
- `retrieval_evaluator.py`: relevance filtering and chunk evaluation.
- `clarification_gen.py`: context-grounded question generation.
- `state_manager.py`: multi-turn state and history management.

### 4.3 `src/coffee_diagnosis/diagnosis`

- `controller.py`: full orchestration of diagnosis flow.
- `diagnosis_generator.py`: final diagnosis synthesis.
- `hallucination_checker.py`: multi-generation consistency checks.

### 4.4 UI

- `ui/streamlit_app.py`: interactive web interface with evidence visibility and stepwise interaction.

## 5. Repository Organization

```text
GenAi project/
├── config/
│   └── settings.py
├── data/
│   ├── disease_knowledge.json
│   ├── disease_question_priorities.json
│   ├── pdfs/
│   ├── vector_db/
│   └── vector_db_json/
├── scripts/
│   ├── restart_app.sh
│   └── diagnose.py
├── src/coffee_diagnosis/
├── test/
│   ├── test_diagnosis.py
│   ├── test_multiturn.py
│   ├── test_hybrid.py
│   ├── evaluate_metrics.py
│   ├── evaluate_v2.py
│   ├── metrics_v2.py
│   ├── run_tests.py
│   ├── test_brown_mites_fix.py
│   ├── validate_phase0.py
│   ├── validate_phase1.py
│   ├── validate_phase2.py
│   ├── test_phase2_metrics.py
│   ├── evaluation_dataset.json
│   ├── evaluation_dataset_v2.json
│   └── TESTING_GUIDE.md
├── ui/
│   └── streamlit_app.py
├── README.md
└── requirements.txt
```
Note: The LaTeX paper and final status artifacts are located in the sister directory `../paper/` parallel to the project root.

## 6. Environment and Setup

### 6.1 Prerequisites

- Python 3.11 (recommended),
- Ollama installed,
- required model available locally (default: `phi3`).

### 6.2 Installation

```bash
pip install -r requirements.txt
```

### 6.3 `.env` (local default)

```bash
LLM_PROVIDER=openai_local
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=phi3
LLM_API_KEY=ollama
LOG_LEVEL=INFO
```

### 6.4 Model Pull

```bash
ollama pull phi3
```

## 7. Running the Application

### 7.1 Streamlit UI

Run the Streamlit application directly:
```bash
python -m streamlit run ui/streamlit_app.py
```
Or use the helper script:
```bash
./scripts/restart_app.sh
```

### 7.2 Running the Unified Test Runner

Run pytest suites with sensible configurations:
```bash
# Run fast unit tests only (no LLM required, ~5s)
python test/run_tests.py --unit

# Run full integration tests (requires local LLM running)
python test/run_tests.py --integration

# Run regression baseline comparison
python test/run_tests.py --regression

# Run Phase 2 metric and schema validations
python test/run_tests.py --validate

# Run all test suites
python test/run_tests.py --all
```

### 7.3 Direct Test/Evaluation Execution

```bash
python test/test_diagnosis.py "Leaves are turning yellow with brown spots"
python test/test_multiturn.py
python test/test_hybrid.py
python test/evaluate_metrics.py   # Baseline evaluation (12 cases)
python test/evaluate_v2.py        # Phase 2 evaluation (55+ cases)
```

## 8. Test Suite Description

### 8.1 `test/test_diagnosis.py`

Smoke-level diagnosis execution for direct symptom input.

### 8.2 `test/test_multiturn.py`

Validates question-answer interaction flow and controller transitions.

### 8.3 `test/test_hybrid.py`

Checks disease-distribution behavior under diverse prompts to detect dominant-disease bias.

### 8.4 `test/evaluate_metrics.py`

Runs the baseline labeled evaluation (12 cases).

### 8.5 `test/evaluate_v2.py`

Runs the expanded labeled evaluation (55+ cases) and calculates detailed metrics including Top-K Accuracy, Mean Reciprocal Rank (MRR), and Multi-label Precision/Recall.

### 8.6 Validation Scripts

- `test/validate_phase0.py`: Validates Phase 0 targeted, priority-based question logic (e.g. Red Spider Mites).
- `test/validate_phase1.py`: Validates Phase 1 expanded evaluation dataset schema and distribution.
- `test/validate_phase2.py`: Validates calculations of precision, recall, and state manager behavior under increased `MAX_QUESTIONS = 5`.
- `test/test_phase2_metrics.py`: Dedicated test suite verifying implementation of MRR, multi-label recall, precision, and Jaccard similarity fallback.

## 9. Current Experimental Results

### 9.1 Labeled Evaluation (Baseline, 12 cases)

- Top-1 accuracy (single-turn): 50.00%
- Top-1 accuracy (multi-turn): 66.67%
- Top-3 accuracy (proxy): 83.33%
- Avg consistency score: 88.89%
- Hallucination rate: 25.00%
- Avg Precision@5 (keyword proxy): 48.19%
- Avg CRAG relevance score: 0.6588
- Top-1 improvement: +16.67%

### 9.2 Expanded Labeled Evaluation (Phase 2, 55+ cases)
The system has been scaled and evaluated against a diverse dataset of 55+ test cases representing complex multi-turn scenarios, achieving:
- High Top-K Accuracy and Mean Reciprocal Rank (MRR)
- Robust multi-label recall and precision during diagnostic candidate comparisons
- Accurate Jaccard similarity fallback logic for sparse keyword situations

### 9.3 Hybrid Distribution Check (`test/test_hybrid.py`, 8 cases)

- no single-disease dominance observed,
- Coffee Leaf Rust frequency: 2/8 (25.0%).

## 10. Paper Artifacts

All manuscript and submission artifacts are under the parallel `../paper/` directory.

Primary manuscript:

`../paper/coffee_disease_diagnosis_ieee.tex`

## 11. Configuration Notes

Primary runtime configuration is in `config/settings.py` and `.env`.

Commonly adjusted values include:
- maximum clarification turns,
- retrieval depth/top-k,
- relevance threshold,
- verification generation count,
- provider/model selection.

## 12. Failure Modes and Troubleshooting

### 12.1 Import or path errors

Run commands from project root and use `test/...` paths (not `tests/...`).

### 12.2 Slow first startup

Initial runs build/load vector indexes. Subsequent runs should reuse cached indexes.

### 12.3 Weak diagnosis quality

Provide symptom details for color, pattern, location, spread, and timing; allow clarification turns.

### 12.4 LLM backend issues

Confirm Ollama service and model availability:

```bash
ollama list
```

## 13. Ethical and Practical Disclaimer

This system is a research prototype for decision support. It should be used with agronomic judgment and local field validation before operational deployment.

## 14. CoffeeGuard API and Operations Platform

The existing Streamlit diagnosis UI and RAG controller remain in place. The API calls the same multi-turn controller and hybrid FAISS/JSON/PDF retrieval, CRAG filtering, diagnosis generation, and hallucination verification; it does not replace the diagnosis pipeline.

### Architecture

- `src/coffee_diagnosis/api/`: FastAPI endpoints, input schemas, per-session orchestration, Prometheus metrics, SQLite incident tracking, threshold monitoring, and support-agent tools.
- `src/coffee_diagnosis/mcp_server.py`: separate stdio MCP server. It exposes diagnosis and support tools and can combine multiple read-only tools for investigations.
- `data/runbooks/`: operational troubleshooting references. Disease evidence continues to come from the existing JSON and PDF retrieval path.
- `deploy/prometheus/` and `deploy/grafana/`: scrape configuration and provisioned operational dashboard.
- `docker-compose.yml`: API, existing Streamlit interface, Prometheus, Grafana, and an optional MCP profile.

The API serializes access to the shared retrieval engine while preserving separate conversation state and clarification history for each session. The controller/model is initialized lazily on the first diagnosis or disease-document search; first-run FAISS model/index initialization can take time.

### Local API setup

Use Python 3.11 and install the project dependencies:

```bash
python -m pip install -r requirements.txt
python -m uvicorn coffee_diagnosis.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

The original UI remains available separately:

```bash
python -m streamlit run ui/streamlit_app.py
```

Useful endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | Required local knowledge-file readiness |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `POST` | `/api/v1/diagnosis` | Start a diagnosis session; body `{"query":"orange powder under coffee leaves"}` |
| `POST` | `/api/v1/diagnosis/{session_id}/answers` | Continue a multi-turn diagnosis; body `{"answer":"mostly older leaves"}` |
| `DELETE` | `/api/v1/diagnosis/{session_id}` | Remove a conversation session |
| `GET` | `/api/v1/telemetry` | Current measured request-window summary |
| `GET` | `/api/v1/errors/recent` | Recent 5xx metadata (no request bodies) |
| `POST` | `/api/v1/knowledge/search` | Search disease documents and operational runbooks |
| `POST` | `/api/v1/support/investigate` | Combine health, errors, knowledge, and incident history |
| `GET/POST` | `/api/v1/incidents` | List or manually create incidents |
| `POST` | `/api/v1/incidents/{id}/acknowledge` | Acknowledge an incident |
| `POST` | `/api/v1/incidents/{id}/resolve` | Record a human-entered resolution |
| `POST` | `/api/v1/slack/commands` | Optional signed Slack slash-command endpoint |

Example diagnosis flow:

```bash
curl -X POST http://localhost:8000/api/v1/diagnosis ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"orange powder under coffee leaves\"}"
```

Continue with the returned `session_id` and the generated question:

```bash
curl -X POST http://localhost:8000/api/v1/diagnosis/SESSION_ID/answers ^
  -H "Content-Type: application/json" ^
  -d "{\"answer\":\"mostly on older leaves\"}"
```

For PowerShell, replace the shell continuation character (`^`) with a backtick or issue the command on one line.

### Environment configuration

Copy `.env.example` to `.env` for Compose interpolation and set a private `GRAFANA_ADMIN_PASSWORD`. Never commit `.env` or Slack/LLM credentials. LLM settings preserve the existing local OpenAI-compatible endpoint defaults; hosted integrations remain optional.

Important configuration includes:

- `LLM_TIMEOUT_SECONDS`: upper bound for each provider request.
- `API_MAX_REQUEST_BYTES`, `API_MAX_SESSIONS`, `API_SESSION_TTL_SECONDS`: API body and session limits.
- `INCIDENT_ERROR_RATE_THRESHOLD`, `INCIDENT_LATENCY_THRESHOLD_SECONDS`, `INCIDENT_WINDOW_SECONDS`, `INCIDENT_MIN_REQUESTS`: measured request SLI alert rules.
- `INCIDENT_DB_PATH`: SQLite incident state path.
- `SUPPORT_LLM_ENABLED`: enable optional support-summary generation; disabled by default. Grounded template summaries remain available without an LLM.
- `SLACK_SIGNING_SECRET`: enables signed commands. `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` additionally enable incident notifications.
- `DIAGNOSIS_EVALUATION_FILE`: optional path to a generated labeled-evaluation result JSON.

### Observability and data provenance

The API exports request counters, error counters, latency histograms, diagnosis outcomes, retrieval outcomes, and current incident counts. Grafana shows request volume, p50/p95/p99 latency, HTTP errors, diagnosis outcomes, retrieval outcomes, and incidents.

Live prediction results are **not** treated as verified labels. Diagnosis quality panels remain absent until `DIAGNOSIS_EVALUATION_FILE` points to a result produced by `python test/evaluate_v2.py`; the evaluator writes `data_source: labeled_evaluation`, the dataset name, evaluation time, and case count into its result. The API rejects result files without that provenance or out-of-range metrics. Run evaluation explicitly:

```bash
python test/evaluate_v2.py --dataset test/evaluation_dataset_v2.json
```

Then set `DIAGNOSIS_EVALUATION_FILE=test/results/evaluation_v2_results.json` and restart the API. With Compose, the results directory is a persistent volume; run the evaluator in the API container with `docker compose exec api python test/evaluate_v2.py --dataset test/evaluation_dataset_v2.json`, set the same path in `.env`, and recreate the API. The evaluator requires a functioning local/hosted model and embedding dependencies. No evaluation result is bundled or generated from live API traffic.

Request-window telemetry and threshold incidents are real observations from requests received by that process. They are labeled `measured_request_telemetry` with `simulated: false`; the platform does not seed fabricated incidents or benchmark values. Prometheus counters are process-local and reset on restart.

### Grafana and Docker Compose

Set the Grafana password as described above, then run:

```bash
docker compose up --build
```

Services:

- API: <http://localhost:8000>
- Existing Streamlit UI: <http://localhost:8501>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (username defaults to `admin`; use the password from `.env`)

The CoffeeGuard dashboard is provisioned automatically. Prometheus starts scraping when the API is available; dashboard panels have no live history until requests have occurred. To start the standalone MCP server in Compose, use `docker compose --profile mcp run --rm mcp`. The model endpoint from a container defaults to `host.docker.internal:11434`; adjust it if Ollama runs elsewhere.

### MCP tools

The stdio MCP server provides `diagnose`, `submit_diagnosis_answer`, `knowledge_search`, `service_health`, `recent_errors`, `incident_history`, and `investigate`. Tools validate bounded inputs, return structured objects, and report unavailable services explicitly. A client can call multiple tools or call `investigate` to combine their results.

Run locally from the project root:

```bash
python -m coffee_diagnosis.mcp_server
```

When using a stdio MCP client, set its working directory to the project and set `PYTHONPATH` to the project's `src` directory.

### Slack commands and notifications

Configure `SLACK_SIGNING_SECRET` to enable `/api/v1/slack/commands`. Slack requests are HMAC signature-checked with a five-minute timestamp window by default, and `trigger_id` is deduplicated in a local SQLite store. Supported slash-command syntax:

- `/coffee diagnose <symptoms>`
- `/coffee health`
- `/coffee runbook <query>`
- `/coffee incidents [query]`

Set a bot token and channel ID to deliver incident notifications with severity, evidence, current status, and dashboard link. Without credentials, the command endpoint is disabled and notifications are skipped. Configure the Slack app to send slash commands to the endpoint URL over HTTPS; use the official Slack signing secret, not a hard-coded token.

### Incident workflows and safety

Incidents are stored in SQLite and can be opened manually or automatically when the configured measured 5xx rate or p95 request latency threshold is reached. Records include the observed window, measured values, request IDs/status metadata, and dashboard link. Operators can acknowledge incidents and add a resolution. There is no remediation execution tool or endpoint: AI summaries are advisory, and operational changes require explicit human approval outside this service.

### Tests, CI, and limitations

Run the fast tests (no live LLM required):

```bash
python -m pytest test/unit test/api -q
```

The GitHub Actions workflow runs these unit/API tests and validates deployment manifests. The existing integration and benchmark suites are still available but require model/dependency setup and can be slow:

```bash
python test/run_tests.py --integration
python test/evaluate_v2.py --dataset test/evaluation_dataset_v2.json
```

This remains a single-process research deployment: session state is in memory, while incident and Slack deduplication records use SQLite. Horizontal scaling requires shared session/telemetry storage and coordination. API authentication, authorization, production secrets management, retention policies, and deployment-specific network controls must be supplied before public deployment. Slack is optional; MCP uses stdio and is intended to be launched by a trusted local MCP host. Diagnosis remains a decision-support aid and should be reviewed by an agronomist.
