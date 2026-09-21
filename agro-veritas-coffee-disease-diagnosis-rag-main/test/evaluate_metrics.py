"""
Evaluate core metrics:
1) Diagnosis accuracy (Top-1, Top-3)
2) Hallucination rate / consistency
3) Retrieval quality (Precision@k proxy + avg CRAG score)
4) Multi-turn improvement (single-turn vs multi-turn)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def run_single_turn(controller: CoffeeDiagnosisController, query: str) -> Dict:
    """
    Single-turn baseline:
    - start_diagnosis(query)
    - if question returned, force immediate minimal response once
    """
    first = controller.start_diagnosis(query)
    if first.get("status") == "diagnosis":
        return first

    # Minimal baseline interaction (one answer), then continue until diagnosis or max loops.
    loops = 0
    cur = first
    while cur.get("status") == "question" and loops < 2:
        cur = controller.submit_answer("not sure")
        loops += 1
    return cur


def run_multi_turn(
    controller: CoffeeDiagnosisController, query: str, followup_answers: List[str]
) -> Dict:
    """
    Multi-turn evaluation mode:
    - answer using labeled followup answers for realism
    - fallback to "not sure" only if answers are exhausted
    """
    cur = controller.start_diagnosis(query)
    loops = 0
    answer_idx = 0
    while cur.get("status") == "question" and loops < 8:
        if answer_idx < len(followup_answers):
            answer = followup_answers[answer_idx]
            answer_idx += 1
        else:
            answer = "not sure"
        cur = controller.submit_answer(answer)
        loops += 1
    return cur


def extract_top3_from_reason(reason: str, expected_list: List[str]) -> List[str]:
    # Lightweight proxy: if candidate names appear in reason text, include them.
    reason_l = normalize(reason)
    found = []
    for d in expected_list:
        if normalize(d) in reason_l:
            found.append(d)
    return found[:3]


def retrieval_metrics_from_context(ctx: List[Dict], keywords: List[str], k: int = 5) -> Dict:
    if not ctx:
        return {"precision_at_k": 0.0, "avg_relevance_score": 0.0}

    top = ctx[:k]
    kw = [normalize(x) for x in keywords]
    hits = 0
    rel_scores = []
    for c in top:
        content = normalize(c.get("content", ""))
        if any(kword in content for kword in kw):
            hits += 1
        rel_scores.append(float(c.get("relevance_score", 0.0)))

    precision_at_k = hits / max(len(top), 1)
    avg_relevance_score = sum(rel_scores) / max(len(rel_scores), 1)
    return {
        "precision_at_k": precision_at_k,
        "avg_relevance_score": avg_relevance_score
    }


def evaluate():
    dataset_path = PROJECT_ROOT / "test" / "evaluation_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    controller = CoffeeDiagnosisController(
        data_dir=str(PROJECT_ROOT / "data" / "pdfs"),
        vector_db_path=str(PROJECT_ROOT / "data" / "vector_db")
    )

    total = len(cases)
    top1_single = 0
    top1_multi = 0
    top3_multi = 0
    hallucinations = 0
    consistency_scores = []
    p_at_k_list = []
    crag_scores = []

    print(f"\nEvaluating {total} labeled cases...\n")

    for case in cases:
        q = case["query"]
        expected = case["expected_disease"]
        top3_expected = case.get("top3_expected", [expected])
        keywords = case.get("relevant_keywords", [])
        followup_answers = case.get("followup_answers", [])

        # Single-turn baseline
        controller.reset()
        r1 = run_single_turn(controller, q)
        d1 = r1.get("diagnosis")
        pred1 = d1.disease_name if d1 else ""
        if normalize(pred1) == normalize(expected):
            top1_single += 1

        # Multi-turn mode
        controller.reset()
        r2 = run_multi_turn(controller, q, followup_answers)
        d2 = r2.get("diagnosis")
        v2 = r2.get("verification", {})
        pred2 = d2.disease_name if d2 else ""

        if normalize(pred2) == normalize(expected):
            top1_multi += 1

        # Top-3 proxy (using expected set + reason text)
        reason = d2.reason if d2 else ""
        found_candidates = set(extract_top3_from_reason(reason, top3_expected))
        if normalize(pred2) in [normalize(x) for x in top3_expected] or found_candidates:
            top3_multi += 1

        # Hallucination metrics
        if v2.get("hallucination_detected", False):
            hallucinations += 1
        consistency_scores.append(float(v2.get("consistency_score", 0.0)))

        # Retrieval metrics
        ctx = controller.state_manager.state.retrieved_context
        rm = retrieval_metrics_from_context(ctx, keywords, k=5)
        p_at_k_list.append(rm["precision_at_k"])
        crag_scores.append(rm["avg_relevance_score"])

    top1_single_acc = top1_single / total
    top1_multi_acc = top1_multi / total
    top3_multi_acc = top3_multi / total
    hallucination_rate = hallucinations / total
    avg_consistency = sum(consistency_scores) / max(len(consistency_scores), 1)
    avg_p_at_5 = sum(p_at_k_list) / max(len(p_at_k_list), 1)
    avg_crag = sum(crag_scores) / max(len(crag_scores), 1)
    improvement = top1_multi_acc - top1_single_acc

    print("=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Total cases: {total}")
    print("\n1) DIAGNOSIS ACCURACY")
    print(f"Top-1 Accuracy (single-turn baseline): {top1_single_acc*100:.2f}%")
    print(f"Top-1 Accuracy (multi-turn):          {top1_multi_acc*100:.2f}%")
    print(f"Top-3 Accuracy (multi-turn, proxy):    {top3_multi_acc*100:.2f}%")

    print("\n2) HALLUCINATION / CONSISTENCY")
    print(f"Avg Consistency Score: {avg_consistency*100:.2f}%")
    print(f"Hallucination Rate:    {hallucination_rate*100:.2f}%")

    print("\n3) RETRIEVAL QUALITY")
    print(f"Avg Precision@5 (keyword proxy): {avg_p_at_5*100:.2f}%")
    print(f"Avg CRAG relevance score:        {avg_crag:.4f}")

    print("\n4) MULTI-TURN IMPROVEMENT")
    print(f"Top-1 improvement: {(improvement*100):+.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    evaluate()

