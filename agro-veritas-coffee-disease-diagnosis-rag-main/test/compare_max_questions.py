"""
Phase 3: A/B Comparison — MAX_QUESTIONS=3 vs MAX_QUESTIONS=5

Runs a representative subset of the evaluation dataset under both limits
and reports the accuracy delta, average questions asked, and time.

Usage:
    python test/compare_max_questions.py

Requires: Local LLM (Ollama) running on http://localhost:11434
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def run_case_with_limit(
    data_dir: str,
    vector_db_path: str,
    query: str,
    followup_answers: List[str],
    max_q: int
) -> Tuple[str, int, float]:
    """
    Run a single test case with a given max_questions limit.

    Returns:
        (predicted_disease, questions_asked, time_seconds)
    """
    from coffee_diagnosis.rag.state_manager import StateManager

    # Patch StateManager for this run with specific max_questions
    from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController

    controller = CoffeeDiagnosisController(
        data_dir=data_dir,
        vector_db_path=vector_db_path
    )
    # Override the state manager's max_questions for A/B
    controller.state_manager._max_questions = max_q

    t0 = time.time()
    cur = controller.start_diagnosis(query)
    loops = 0
    answer_idx = 0
    while cur.get("status") == "question" and loops < (max_q + 2):
        answer = followup_answers[answer_idx] if answer_idx < len(followup_answers) else "not sure"
        answer_idx += 1
        cur = controller.submit_answer(answer)
        loops += 1

    elapsed = time.time() - t0
    d = cur.get("diagnosis")
    pred = d.disease_name if d else "Unknown"
    qs_asked = controller.state_manager.state.questions_asked
    return pred, qs_asked, elapsed


def load_representative_cases(dataset_path: Path, per_type: int = 4) -> List[Dict]:
    """
    Select a balanced subset: per_type cases from each disease category.
    Prioritizes multi-symptom diseases where extra questions matter most.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])

    # Group by disease
    by_disease: Dict[str, List] = {}
    for c in cases:
        d = c.get("expected_disease", "Unknown")
        by_disease.setdefault(d, []).append(c)

    # Prioritize diseases that benefit from more questions (multi-symptom / nutrient deficiencies)
    priority_diseases = [
        "Red Spider Mites", "Coffee Berry Borer", "Root Rot / Wilt Disease",
        "Coffee Wilt Disease", "Magnesium Deficiency", "Nitrogen Deficiency",
        "Iron Deficiency", "Potassium Deficiency"
    ]

    selected = []
    # Priority diseases first
    for disease in priority_diseases:
        if disease in by_disease:
            selected.extend(by_disease[disease][:per_type])
    # Fill remaining from other diseases
    for disease, disease_cases in by_disease.items():
        if disease not in priority_diseases:
            selected.extend(disease_cases[:2])

    return selected[:30]  # Cap at 30 for reasonable runtime


def compare(data_dir: str, vector_db_path: str, dataset_path: Path) -> None:
    cases = load_representative_cases(dataset_path)
    total = len(cases)

    print(f"\n{'='*72}")
    print("PHASE 3: A/B COMPARISON — MAX_QUESTIONS=3 vs MAX_QUESTIONS=5")
    print(f"{'='*72}")
    print(f"Running {total} representative cases...\n")

    results = {3: [], 5: []}

    for i, case in enumerate(cases, 1):
        q = case["query"]
        expected = case["expected_disease"]
        followup = case.get("followup_answers", [])

        print(f"  [{i:2d}/{total}] {expected[:40]:<40} | ", end="", flush=True)

        for max_q in [3, 5]:
            pred, qs, t = run_case_with_limit(data_dir, vector_db_path, q, followup, max_q)
            correct = normalize(pred) == normalize(expected)
            results[max_q].append({
                "disease": expected,
                "predicted": pred,
                "correct": correct,
                "questions": qs,
                "time": t
            })

        r3 = results[3][-1]
        r5 = results[5][-1]
        print(
            f"Q=3: {'✅' if r3['correct'] else '❌'}({r3['questions']}q)  "
            f"Q=5: {'✅' if r5['correct'] else '❌'}({r5['questions']}q)"
        )

    # Aggregate
    def agg(res_list):
        correct = sum(1 for r in res_list if r["correct"])
        avg_q = sum(r["questions"] for r in res_list) / len(res_list)
        avg_t = sum(r["time"] for r in res_list) / len(res_list)
        return correct / len(res_list), avg_q, avg_t

    acc3, avg_q3, avg_t3 = agg(results[3])
    acc5, avg_q5, avg_t5 = agg(results[5])

    # Per-disease breakdown for priority diseases
    print(f"\n{'='*72}")
    print("PER-DISEASE BREAKDOWN (Priority multi-symptom diseases)")
    print(f"{'='*72}")
    print(f"{'Disease':<35} {'Q=3 Acc':>8} {'Q=5 Acc':>8} {'Δ':>6} {'Q=3 avg':>7} {'Q=5 avg':>7}")
    print("-" * 72)

    all_diseases = sorted(set(r["disease"] for r in results[3]))
    for disease in all_diseases:
        r3_d = [r for r in results[3] if r["disease"] == disease]
        r5_d = [r for r in results[5] if r["disease"] == disease]
        if not r3_d:
            continue
        a3 = sum(1 for r in r3_d if r["correct"]) / len(r3_d)
        a5 = sum(1 for r in r5_d if r["correct"]) / len(r5_d)
        q3 = sum(r["questions"] for r in r3_d) / len(r3_d)
        q5 = sum(r["questions"] for r in r5_d) / len(r5_d)
        delta = (a5 - a3) * 100
        delta_str = f"+{delta:.0f}%" if delta > 0 else (f"{delta:.0f}%" if delta < 0 else " 0%")
        print(f"{disease[:35]:<35} {a3*100:>7.1f}% {a5*100:>7.1f}% {delta_str:>6} {q3:>7.1f} {q5:>7.1f}")

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"{'Metric':<30} {'MAX_Q=3':>12} {'MAX_Q=5':>12} {'Delta':>10}")
    print("-" * 65)
    print(f"{'Top-1 Accuracy':<30} {acc3*100:>11.2f}% {acc5*100:>11.2f}% {(acc5-acc3)*100:>+9.2f}%")
    print(f"{'Avg Questions Asked':<30} {avg_q3:>12.2f} {avg_q5:>12.2f} {avg_q5-avg_q3:>+10.2f}")
    print(f"{'Avg Time per Case (s)':<30} {avg_t3:>12.2f} {avg_t5:>12.2f} {avg_t5-avg_t3:>+10.2f}")
    print(f"{'='*72}")

    verdict = "✅ MAX_QUESTIONS=5 IMPROVES ACCURACY" if acc5 > acc3 else (
        "➡️  NO ACCURACY CHANGE (early-stopping works well)" if acc5 == acc3 else
        "⚠️  MAX_QUESTIONS=5 SLIGHTLY WORSE (check early-stopping logic)"
    )
    print(f"\nVERDICT: {verdict}")
    print(f"{'='*72}\n")

    # Save results
    results_dir = PROJECT_ROOT / "test" / "results"
    results_dir.mkdir(exist_ok=True)
    out = {
        "total_cases": total,
        "max_q_3": {"accuracy": acc3, "avg_questions": avg_q3, "avg_time": avg_t3},
        "max_q_5": {"accuracy": acc5, "avg_questions": avg_q5, "avg_time": avg_t5},
        "delta_accuracy": acc5 - acc3,
        "per_case_q3": results[3],
        "per_case_q5": results[5]
    }
    out_path = results_dir / "compare_max_questions.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}\n")


if __name__ == "__main__":
    data_dir = str(PROJECT_ROOT / "data" / "pdfs")
    vector_db_path = str(PROJECT_ROOT / "data" / "vector_db")
    dataset_path = PROJECT_ROOT / "test" / "evaluation_dataset_v2.json"

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    compare(data_dir, vector_db_path, dataset_path)
