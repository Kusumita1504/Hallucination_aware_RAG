"""
Phase 2: Validate Metric Functions (No LLM Required)
======================================================
Unit-level validation of all Phase 2 metric functions using fixed synthetic examples.
Fast to run as a pre-commit check — no network, no LLM, no GPU needed.

Usage:
    python test/validate_phase2.py
"""

import sys
import json
import re
import math
from pathlib import Path
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def normalize(text: str) -> str:
    return (text or "").strip().lower()


# ─── Metric implementations (copied from evaluate_v2.py for isolated testing) ──

def extract_candidates(response_text: str) -> List[Tuple[str, float]]:
    candidates = []
    match = re.search(
        r'CANDIDATE\s+COMPARISON:?(.*?)(?:FINAL\s+DIAGNOSIS|PRIMARY\s+DIAGNOSIS|$)',
        response_text, re.DOTALL | re.IGNORECASE
    )
    if not match:
        return candidates
    section = match.group(1)
    for line in section.split('\n'):
        m = re.search(r'\d+\.\s*([^:]+):\s*(\d+(?:\.\d+)?)\s*%', line)
        if m:
            candidates.append((m.group(1).strip(), float(m.group(2)) / 100.0))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def top_k_hit(candidates: List[Tuple[str, float]], expected: str, k: int) -> bool:
    expected_n = normalize(expected)
    for disease, _ in candidates[:k]:
        if normalize(disease) == expected_n:
            return True
    return False


def mrr(candidates: List[Tuple[str, float]], expected: str) -> float:
    expected_n = normalize(expected)
    for i, (disease, _) in enumerate(candidates, 1):
        if normalize(disease) == expected_n:
            return 1.0 / i
    return 0.0


def multi_label_precision(candidates: List[Tuple[str, float]], plausible: List[str], k: int) -> float:
    if not candidates:
        return 0.0
    pn = [normalize(d) for d in plausible]
    hits = sum(1 for d, _ in candidates[:k] if normalize(d) in pn)
    return hits / min(k, len(candidates))


def multi_label_recall(candidates: List[Tuple[str, float]], plausible: List[str], k: int) -> float:
    if not plausible:
        return 1.0
    pn = [normalize(d) for d in plausible]
    pred_n = [normalize(d) for d, _ in candidates[:k]]
    return sum(1 for p in pn if p in pred_n) / len(plausible)


def jaccard_sim(text_a: str, text_b: str) -> float:
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─── Test functions ─────────────────────────────────────────────────────────────

def check(condition: bool, name: str, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    sep = " - " if detail else ""
    print(f"  {icon} {name}{sep}{detail}")
    return condition


def test_candidate_extraction() -> int:
    """Tests that CANDIDATE COMPARISON block is correctly parsed."""
    print("\n[TEST] Candidate extraction from LLM output")
    errors = 0

    sample = """
CANDIDATE COMPARISON:
1. Coffee Leaf Rust: 85% - Orange powder on underside of leaves
2. Brown Eye Spot: 10% - No gray center observed
3. Phoma Leaf Spot: 5% - Unlikely given edge pattern

FINAL DIAGNOSIS:
DISEASE: Coffee Leaf Rust
"""
    cands = extract_candidates(sample)
    errors += not check(len(cands) == 3, "Extracts 3 candidates", f"got {len(cands)}")
    errors += not check(normalize(cands[0][0]) == "coffee leaf rust",
                         "First candidate is Coffee Leaf Rust", cands[0][0])
    errors += not check(abs(cands[0][1] - 0.85) < 0.01,
                         "First confidence is 0.85", str(cands[0][1]))
    errors += not check(cands[0][1] >= cands[1][1] >= cands[2][1],
                         "Sorted descending by confidence")

    # Edge cases
    empty_cands = extract_candidates("No candidate section here.")
    errors += not check(len(empty_cands) == 0, "Empty input returns empty list")

    decimal_sample = """
CANDIDATE COMPARISON:
1. Red Spider Mites: 87.5% - Bronze speckling visible
2. Nitrogen Deficiency: 12.5% - Possible
FINAL DIAGNOSIS:
"""
    d_cands = extract_candidates(decimal_sample)
    errors += not check(len(d_cands) == 2, "Handles decimal percentages")
    errors += not check(abs(d_cands[0][1] - 0.875) < 0.01,
                         "Decimal confidence 87.5% → 0.875", str(d_cands[0][1]))

    return errors


def test_top_k_accuracy() -> int:
    """Tests Top-1, Top-3, and Top-5 accuracy."""
    print("\n[TEST] Top-K accuracy")
    errors = 0

    candidates = [
        ("Coffee Leaf Rust", 0.85),
        ("Brown Eye Spot", 0.10),
        ("Phoma Leaf Spot", 0.05),
    ]

    errors += not check(top_k_hit(candidates, "Coffee Leaf Rust", k=1), "Top-1 hit for rank-1 disease")
    errors += not check(not top_k_hit(candidates, "Brown Eye Spot", k=1), "Top-1 miss for rank-2 disease")
    errors += not check(top_k_hit(candidates, "Brown Eye Spot", k=3), "Top-3 hit for rank-2 disease")
    errors += not check(top_k_hit(candidates, "Phoma Leaf Spot", k=3), "Top-3 hit for rank-3 disease")
    errors += not check(not top_k_hit(candidates, "Unknown Disease", k=3), "Top-3 miss for absent disease")

    # Case insensitivity
    errors += not check(top_k_hit(candidates, "coffee leaf rust", k=1), "Case-insensitive Top-1")
    errors += not check(top_k_hit(candidates, "BROWN EYE SPOT", k=3), "Case-insensitive Top-3")

    # Single candidate fallback
    single = [("Root Rot / Wilt Disease", 0.9)]
    errors += not check(top_k_hit(single, "Root Rot / Wilt Disease", k=1), "Single candidate Top-1")

    return errors


def test_mrr() -> int:
    """Tests Mean Reciprocal Rank calculation."""
    print("\n[TEST] Mean Reciprocal Rank (MRR)")
    errors = 0

    candidates = [
        ("Coffee Leaf Rust", 0.70),
        ("Brown Eye Spot", 0.20),
        ("Phoma Leaf Spot", 0.10),
    ]

    mrr1 = mrr(candidates, "Coffee Leaf Rust")
    errors += not check(mrr1 == 1.0, "MRR=1.0 when rank-1 is correct", str(mrr1))

    mrr2 = mrr(candidates, "Brown Eye Spot")
    errors += not check(abs(mrr2 - 0.5) < 0.001, "MRR=0.5 when rank-2 is correct", str(mrr2))

    mrr3 = mrr(candidates, "Phoma Leaf Spot")
    errors += not check(abs(mrr3 - 1/3) < 0.001, "MRR=0.333 when rank-3 is correct", str(mrr3))

    mrr_none = mrr(candidates, "Anthracnose")
    errors += not check(mrr_none == 0.0, "MRR=0.0 when disease not in list", str(mrr_none))

    # Empty candidates
    mrr_empty = mrr([], "Coffee Leaf Rust")
    errors += not check(mrr_empty == 0.0, "MRR=0.0 for empty candidate list")

    return errors


def test_multi_label_precision() -> int:
    """Tests multi-label precision@k for ambiguous cases."""
    print("\n[TEST] Multi-label Precision@K")
    errors = 0

    candidates = [
        ("Coffee Leaf Rust", 0.60),
        ("Brown Eye Spot", 0.30),
        ("Phoma Leaf Spot", 0.10),
    ]
    plausible = ["Coffee Leaf Rust", "Brown Eye Spot"]

    p2 = multi_label_precision(candidates, plausible, k=2)
    errors += not check(p2 == 1.0, "P@2 = 1.0 when both top-2 are plausible", str(p2))

    p3 = multi_label_precision(candidates, plausible, k=3)
    errors += not check(abs(p3 - 2/3) < 0.001, "P@3 = 0.667 when 2/3 are plausible", str(p3))

    p1 = multi_label_precision(candidates, plausible, k=1)
    errors += not check(p1 == 1.0, "P@1 = 1.0 when rank-1 is in plausible", str(p1))

    # Empty candidates
    p_empty = multi_label_precision([], plausible, k=3)
    errors += not check(p_empty == 0.0, "P@3 = 0.0 for empty candidate list")

    return errors


def test_multi_label_recall() -> int:
    """Tests multi-label recall@k for ambiguous cases."""
    print("\n[TEST] Multi-label Recall@K")
    errors = 0

    candidates = [
        ("Coffee Leaf Rust", 0.60),
        ("Brown Eye Spot", 0.25),
        ("Red Spider Mites", 0.10),
        ("Anthracnose", 0.05),
    ]
    plausible = ["Coffee Leaf Rust", "Brown Eye Spot", "Anthracnose"]

    r3 = multi_label_recall(candidates, plausible, k=3)
    errors += not check(abs(r3 - 2/3) < 0.001, "Recall@3 = 0.667 (2/3 plausible found)", str(r3))

    r4 = multi_label_recall(candidates, plausible, k=4)
    errors += not check(r4 == 1.0, "Recall@4 = 1.0 (all 3 plausible found)", str(r4))

    r1 = multi_label_recall(candidates, plausible, k=1)
    errors += not check(abs(r1 - 1/3) < 0.001, "Recall@1 = 0.333 (1/3 plausible found)", str(r1))

    # Empty plausible set → recall is undefined, should return 1.0 (trivially satisfied)
    r_empty = multi_label_recall(candidates, [], k=3)
    errors += not check(r_empty == 1.0, "Recall = 1.0 when plausible list is empty")

    return errors


def test_jaccard_fallback() -> int:
    """Tests Jaccard keyword overlap fallback for retrieval."""
    print("\n[TEST] Jaccard keyword similarity (retrieval fallback)")
    errors = 0

    q = "orange powder on underside of coffee leaves"
    relevant = "Coffee leaf rust shows orange powdery spores on leaf underside"
    irrelevant = "The weather today is sunny and warm"

    j_rel = jaccard_sim(q, relevant)
    j_irr = jaccard_sim(q, irrelevant)

    errors += not check(j_rel > j_irr, "Relevant doc has higher Jaccard than irrelevant",
                         f"{j_rel:.3f} vs {j_irr:.3f}")
    errors += not check(j_rel > 0.1, "Relevant doc Jaccard > 0.1", str(round(j_rel, 3)))
    errors += not check(j_irr < 0.1, "Irrelevant doc Jaccard < 0.1", str(round(j_irr, 3)))

    # Perfect match
    perfect = jaccard_sim("hello world", "hello world")
    errors += not check(perfect == 1.0, "Identical strings → Jaccard = 1.0")

    return errors


def test_dataset_v2_schema() -> int:
    """Validates evaluation_dataset_v2.json has all required fields."""
    print("\n[TEST] evaluation_dataset_v2.json schema validation")
    errors = 0

    dataset_path = PROJECT_ROOT / "test" / "evaluation_dataset_v2.json"
    if not dataset_path.exists():
        print(f"  {WARN} Dataset not found at {dataset_path} — skipping")
        return 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])

    errors += not check(len(cases) >= 50, f"≥50 cases exist", f"got {len(cases)}")

    required_fields = ["id", "query", "expected_disease", "relevant_keywords", "followup_answers"]
    for case in cases:
        for field in required_fields:
            if field not in case:
                print(f"  {FAIL} Case '{case.get('id', '?')}' missing field '{field}'")
                errors += 1
                break

    ambiguous = [c for c in cases if c.get("disambiguation_required", False)]
    errors += not check(len(ambiguous) >= 8, f"≥8 ambiguous cases", f"got {len(ambiguous)}")

    # Check all diseases have ≥2 cases
    disease_counts: Dict[str, int] = {}
    for c in cases:
        d = c.get("expected_disease", "Unknown")
        disease_counts[d] = disease_counts.get(d, 0) + 1

    under_represented = [d for d, n in disease_counts.items()
                         if n < 2 and d != "AMBIGUOUS"]
    if under_represented:
        print(f"  {WARN} Under-represented diseases (< 2 cases): {under_represented}")

    errors += not check(
        len(disease_counts) >= 14,
        f"≥14 unique diseases covered",
        f"got {len(disease_counts)}"
    )

    return errors


def test_state_manager() -> int:
    """Tests StateManager with new configurable max_questions."""
    print("\n[TEST] StateManager — configurable max_questions")
    errors = 0

    try:
        from coffee_diagnosis.rag.state_manager import StateManager

        sm = StateManager(max_questions=3)
        errors += not check(sm.max_questions == 3, "Custom max_questions=3 respected")

        sm_default = StateManager()
        from config import settings
        errors += not check(
            sm_default.max_questions == settings.MAX_QUESTIONS,
            f"Default max_questions = settings.MAX_QUESTIONS ({settings.MAX_QUESTIONS})"
        )

        # Test should_stop at hard cap
        sm3 = StateManager(max_questions=3)
        sm3.initialize("test query")
        sm3.state.questions_asked = 3
        sm3.state.detected_ambiguities = {"missing": {"color": "unknown"}}  # Still has info
        errors += not check(sm3.should_stop(), "should_stop() at hard cap (questions=3)")

        # Test early stop at high confidence
        sm_conf = StateManager(max_questions=5)
        sm_conf.initialize("test query")
        sm_conf.state.confidence = 0.85
        errors += not check(sm_conf.should_stop(), "should_stop() with confidence > 0.8")

        # Test does NOT stop below cap with low confidence
        sm_cont = StateManager(max_questions=5)
        sm_cont.initialize("test query")
        sm_cont.state.questions_asked = 2
        sm_cont.state.confidence = 0.5
        sm_cont.state.detected_ambiguities = {"missing": {"color": "unknown"}}
        errors += not check(not sm_cont.should_stop(),
                             "should_stop() = False below cap with low confidence")

        # Test get_state_summary includes max_questions
        summary = sm_conf.get_state_summary()
        errors += not check("max_questions" in summary,
                             "get_state_summary() includes max_questions key")

    except ImportError as e:
        print(f"  {WARN} StateManager import failed: {e}")
        errors += 1

    return errors


def test_settings_max_questions() -> int:
    """Tests that settings.py has the updated MAX_QUESTIONS=5."""
    print("\n[TEST] Settings — MAX_QUESTIONS updated to 5")
    errors = 0

    try:
        from config import settings
        errors += not check(settings.MAX_QUESTIONS == 5,
                             f"MAX_QUESTIONS = 5", f"got {settings.MAX_QUESTIONS}")
    except ImportError as e:
        print(f"  {WARN} settings import failed: {e}")
        errors += 1

    return errors


# ─── Main runner ────────────────────────────────────────────────────────────────

def main():
    print("="*72)
    print("PHASE 2: Metric Validation (No LLM Required)")
    print("="*72)

    total_errors = 0
    tests = [
        ("Settings: MAX_QUESTIONS=5", test_settings_max_questions),
        ("Candidate extraction", test_candidate_extraction),
        ("Top-K accuracy", test_top_k_accuracy),
        ("Mean Reciprocal Rank", test_mrr),
        ("Multi-label Precision@K", test_multi_label_precision),
        ("Multi-label Recall@K", test_multi_label_recall),
        ("Jaccard fallback", test_jaccard_fallback),
        ("Dataset v2 schema", test_dataset_v2_schema),
        ("StateManager max_questions", test_state_manager),
    ]

    passed = 0
    for name, fn in tests:
        errors = fn()
        total_errors += errors
        if errors == 0:
            passed += 1

    print(f"\n{'='*72}")
    print(f"RESULTS: {passed}/{len(tests)} test groups passed")
    if total_errors == 0:
        print("[OK] ALL PHASE 2 VALIDATION CHECKS PASSED")
    else:
        print(f"[FAIL] {total_errors} assertion(s) failed - see above for details")
    print(f"{'='*72}\n")

    return total_errors


if __name__ == "__main__":
    errors = main()
    sys.exit(0 if errors == 0 else 1)
