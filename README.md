# CoffeeGuard AI

**A hallucination-aware RAG platform for coffee disease diagnosis, with built-in observability, MCP-based incident triage, and a Slack bot for self-service.**

---

## Overview

Farmers and agronomists need fast, trustworthy answers about coffee plant diseases. Plain LLM chatbots can sound confident and still be wrong. CoffeeGuard AI addresses this in two ways:

1. **Trustworthy answers.** A hybrid RAG pipeline grounds every diagnosis in agricultural data and documents, then checks its own output for hallucinations.
2. **Operable like a real service.** A metrics dashboard, an AI incident-triage agent, and a Slack bot make the system easy to monitor, debug, and use without digging through logs.

The RAG pipeline builds on my research paper, *Agro-Veritas: Hallucination-Aware Coffee Disease Diagnosis using Retrieval-Augmented Generation* (submitted to ICECCE 2026, Kuala Lumpur, Malaysia; under review). This project extends that work into a deployable platform with monitoring, incident triage, and Slack workflows.

---

## Key Features

**Hybrid RAG diagnosis**
Retrieves from structured agricultural data and unstructured PDFs using FAISS and Sentence Transformers. It asks multi-turn clarifying questions about symptoms before giving a diagnosis.

**Hallucination checking**
Corrective RAG (CRAG) re-retrieves when the evidence is weak, and a cross-response consistency check catches unsupported claims.

**Observability dashboard**
Tracks service health, request latency, error rates, diagnosis metrics, and incident trends using FastAPI, Prometheus, Grafana, and Streamlit. Issues can be spotted without reading raw logs.

**AI-assisted incident triage (MCP)**
An agent with MCP-based tools retrieves relevant documentation, investigates service errors, and generates evidence-backed troubleshooting summaries, reducing manual debugging effort.

**Slack bot**
Users can run diagnosis queries, check service health, and get incident summaries directly in Slack, without opening separate dashboards.

**Containerized deployment**
The full stack runs with Docker and Docker Compose.


### How a diagnosis works

1. The user describes symptoms through the API, dashboard, or Slack.
2. If the symptoms are ambiguous, the system asks follow-up questions.
3. It retrieves relevant evidence from structured data and PDFs.
4. The LLM drafts a diagnosis grounded in that evidence.
5. CRAG and the consistency check validate the answer, and the system re-retrieves if the evidence is weak.
6. Latency, errors, and diagnosis metrics are exported to Prometheus.

---

## Results

| Metric | Result |
|---|---|
| Top-1 diagnostic accuracy | 66.67% |
| Top-3 diagnostic accuracy | 83.33% |
| Cross-response consistency | 88.89% |
| Retrieval Precision@5 | 48.19% |

Top-1 and Top-3 accuracy come from multi-turn symptom clarification. Retrieval Precision@5 is reported as measured, and improving retrieval quality is the main item under future work.

---

## Tech Stack

- **Language and API:** Python, FastAPI
- **Retrieval:** FAISS, Sentence Transformers, CRAG
- **Agents:** MCP-based tool-calling
- **Observability:** Prometheus, Grafana, Streamlit
- **Integrations:** Slack API
- **Deployment:** Docker, Docker Compose

---

## Getting Started

**Prerequisites:** Docker, Docker Compose, an LLM API key, and Slack app credentials.

```bash
# 1. Copy the example environment file and fill in your keys
cp .env.example .env

# 2. Build and start all services
docker compose up --build
```

Once running, the API, Streamlit dashboard, Prometheus, and Grafana are available on the ports defined in `docker-compose.yml`.

---

## Design Decisions

- **Hybrid retrieval.** Structured data gives precise facts such as symptoms and treatments, while PDFs add richer context. Combining them covers both.
- **Multi-turn clarification.** Many symptoms overlap across diseases, so asking follow-up questions improves accuracy over answering from the first message.
- **CRAG and consistency checks.** A wrong diagnosis is costly, so the system verifies its answers and retrieves again when the evidence is weak.
- **Observability from the start.** Treating the model as a service with latency, error, and quality metrics makes problems visible before users report them.
- **MCP for triage.** Exposing docs, logs, and metrics as tools lets the agent gather evidence itself instead of guessing.
- **Slack interface.** Meeting users where they already work lowers the barrier to using the tools.

---

## Limitations

- Diagnoses are decision support, not a replacement for an agronomist.
- Coverage is limited to the diseases represented in the dataset.
- Retrieval precision has room for improvement.

## Future Work

- Improve retrieval precision with better chunking, re-ranking, and hybrid lexical plus semantic search.
- Evaluate on a larger and more diverse test set.
- Add alerting rules that notify Slack automatically.

---

## Author

**Yerragunta Kusumita**
B.Tech in Computer Science and Engineering, PES University, Bangalore
