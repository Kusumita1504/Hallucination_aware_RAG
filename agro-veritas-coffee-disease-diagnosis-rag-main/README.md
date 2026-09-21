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

