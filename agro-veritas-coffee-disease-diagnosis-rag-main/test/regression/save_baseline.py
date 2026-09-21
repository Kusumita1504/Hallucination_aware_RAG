"""
Generate baseline_v1.json by running the current system on the original 12-case dataset.
This creates a saved reference point for regression testing.

Usage:
    python test/regression/save_baseline.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def generate_baseline() -> dict:
    dataset_path = PROJECT_ROOT / "test" / "evaluation_dataset.json"
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    print(f"\n{'='*60}")
    print("GENERATING REGRESSION BASELINE (v1 dataset)")
    print(f"Cases: {len(cases)}")
    print(f"{'='*60}\n")

    controller = CoffeeDiagnosisController(
        data_dir=str(PROJECT_ROOT / "data" / "pdfs"),
        vector_db_path=str(PROJECT_ROOT / "data" / "vector_db")
    )

    top1_single = 0
    top1_multi = 0
    hallucinations = 0
    times = []
    per_case_results = []

    for i, case in enumerate(cases, 1):
        q = case["query"]
        expected = case["expected_disease"]
        followup = case.get("followup_answers", [])

        print(f"  [{i:2d}/{len(cases)}] {expected}...")
        t0 = time.time()

        # Single-turn
        controller.reset()
        cur = controller.start_diagnosis(q)
        loops = 0
        while cur.get("status") == "question" and loops < 2:
            cur = controller.submit_answer("not sure")
            loops += 1
        d_single = cur.get("diagnosis")
        pred_single = d_single.disease_name if d_single else ""
        single_correct = normalize(pred_single) == normalize(expected)
        if single_correct:
            top1_single += 1

        # Multi-turn
        controller.reset()
        cur = controller.start_diagnosis(q)
        loops = 0
        answer_idx = 0
        while cur.get("status") == "question" and loops < 10:
            ans = followup[answer_idx] if answer_idx < len(followup) else "not sure"
            answer_idx += 1
            cur = controller.submit_answer(ans)
            loops += 1
        d_multi = cur.get("diagnosis")
        v = cur.get("verification", {})
        pred_multi = d_multi.disease_name if d_multi else ""
        multi_correct = normalize(pred_multi) == normalize(expected)
        if multi_correct:
            top1_multi += 1
        if v.get("hallucination_detected", False):
            hallucinations += 1

        elapsed = time.time() - t0
        times.append(elapsed)
        per_case_results.append({
            "case_id": case.get("id", f"case_{i}"),
            "expected_disease": expected,
            "predicted_single": pred_single,
            "predicted_multi": pred_multi,
            "single_correct": single_correct,
            "multi_correct": multi_correct,
            "hallucination": v.get("hallucination_detected", False),
            "consistency_score": v.get("consistency_score", 0.0),
            "time_s": elapsed
        })

        status = "✅" if multi_correct else "❌"
        print(f"         {status} Predicted: {pred_multi}")

    total = len(cases)
    baseline = {
        "version": "v1",
        "total_cases": total,
        "top1_single_accuracy": top1_single / total,
        "top1_multi_accuracy": top1_multi / total,
        "hallucination_rate": hallucinations / total,
        "avg_time_s": sum(times) / len(times),
        "generated_with_max_questions": 5,  # current setting
        "per_case": per_case_results
    }

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Top-1 (single-turn): {baseline['top1_single_accuracy']*100:.2f}%")
    print(f"Top-1 (multi-turn):  {baseline['top1_multi_accuracy']*100:.2f}%")
    print(f"Hallucination rate:  {baseline['hallucination_rate']*100:.2f}%")
    print(f"{'='*60}\n")

    return baseline


if __name__ == "__main__":
    baseline = generate_baseline()

    out_path = PROJECT_ROOT / "test" / "results" / "baseline_v1.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"✅ Baseline saved to {out_path}")
    print("   You can now run regression tests:")
    print("   pytest test/regression/ -v -m regression\n")
