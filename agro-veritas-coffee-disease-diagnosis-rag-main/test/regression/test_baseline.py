"""
Regression tests against the known baseline.

Loads baseline_v1.json (original 12-case results) and asserts:
- Top-1 accuracy has not degraded
- Hallucination rate has not increased
- All diseases from v1 still produce plausible diagnoses

Run with:
    pytest test/regression/ -v -m regression
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

pytestmark = [pytest.mark.regression]

BASELINE_PATH = PROJECT_ROOT / "test" / "results" / "baseline_v1.json"
DATASET_V1_PATH = PROJECT_ROOT / "test" / "evaluation_dataset.json"


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def baseline() -> Dict:
    if not BASELINE_PATH.exists():
        pytest.skip(
            f"Baseline not found at {BASELINE_PATH}. "
            "Run: python test/regression/save_baseline.py to generate it."
        )
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def v1_dataset() -> List[Dict]:
    if not DATASET_V1_PATH.exists():
        pytest.skip(f"Dataset v1 not found at {DATASET_V1_PATH}")
    with open(DATASET_V1_PATH, "r") as f:
        return json.load(f)["cases"]


# ─── Baseline Integrity Tests ────────────────────────────────────────────────────

class TestBaselineStructure:
    def test_baseline_has_accuracy(self, baseline):
        assert "top1_multi_accuracy" in baseline, "Baseline missing top1_multi_accuracy"

    def test_baseline_has_hallucination_rate(self, baseline):
        assert "hallucination_rate" in baseline, "Baseline missing hallucination_rate"

    def test_baseline_has_at_least_12_cases(self, baseline):
        assert baseline.get("total_cases", 0) >= 12, (
            f"Baseline should have ≥ 12 cases, got {baseline.get('total_cases')}"
        )

    def test_baseline_accuracy_was_reasonable(self, baseline):
        acc = baseline["top1_multi_accuracy"]
        assert acc >= 0.5, (
            f"Baseline accuracy was {acc*100:.1f}% — seems too low, check baseline generation"
        )


# ─── Regression Tests (require LLM for current system run) ──────────────────────

class TestNoAccuracyRegression:
    """
    These tests load the baseline and compare against a fresh run.
    They are marked slow because they need the controller.
    """

    @pytest.mark.slow
    @pytest.mark.timeout(600)  # 10 minutes for 12 cases
    def test_current_accuracy_not_worse_than_baseline(self, baseline, v1_dataset, project_root):
        """
        Run all 12 v1 cases with the current system and assert accuracy ≥ baseline.
        """
        from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController

        try:
            controller = CoffeeDiagnosisController(
                data_dir=str(project_root / "data" / "pdfs"),
                vector_db_path=str(project_root / "data" / "vector_db")
            )
        except Exception as e:
            pytest.skip(f"Controller unavailable: {e}")

        def normalize(t):
            return (t or "").strip().lower()

        correct = 0
        total = len(v1_dataset)

        for case in v1_dataset:
            controller.reset()
            q = case["query"]
            expected = case["expected_disease"]
            followup = case.get("followup_answers", [])

            cur = controller.start_diagnosis(q)
            loops = 0
            answer_idx = 0
            while cur.get("status") == "question" and loops < 10:
                ans = followup[answer_idx] if answer_idx < len(followup) else "not sure"
                answer_idx += 1
                cur = controller.submit_answer(ans)
                loops += 1

            d = cur.get("diagnosis")
            pred = d.disease_name if d else ""
            if normalize(pred) == normalize(expected):
                correct += 1

        current_accuracy = correct / total
        baseline_accuracy = baseline["top1_multi_accuracy"]

        # Allow 5% tolerance for LLM non-determinism
        assert current_accuracy >= baseline_accuracy - 0.05, (
            f"Accuracy REGRESSED: current={current_accuracy*100:.1f}% vs "
            f"baseline={baseline_accuracy*100:.1f}% (tolerance: 5%)"
        )

        print(
            f"\nRegression check: {current_accuracy*100:.1f}% vs "
            f"baseline {baseline_accuracy*100:.1f}% — {'✅ PASS' if current_accuracy >= baseline_accuracy - 0.05 else '❌ FAIL'}"
        )


# ─── Hallucination Regression Tests ─────────────────────────────────────────────

class TestNoHallucinationRegression:
    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_hallucination_rate_not_worse(self, baseline, v1_dataset, project_root):
        """
        Hallucination rate (self-consistency) must not be worse than baseline.
        Allows 10% tolerance for LLM non-determinism.
        """
        from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController

        try:
            controller = CoffeeDiagnosisController(
                data_dir=str(project_root / "data" / "pdfs"),
                vector_db_path=str(project_root / "data" / "vector_db")
            )
        except Exception as e:
            pytest.skip(f"Controller unavailable: {e}")

        hallucinations = 0
        total = len(v1_dataset)

        for case in v1_dataset:
            controller.reset()
            cur = controller.start_diagnosis(case["query"])
            loops = 0
            while cur.get("status") == "question" and loops < 10:
                cur = controller.submit_answer("not sure")
                loops += 1

            v = cur.get("verification", {})
            if v.get("hallucination_detected", False):
                hallucinations += 1

        current_rate = hallucinations / total
        baseline_rate = baseline.get("hallucination_rate", 1.0)

        assert current_rate <= baseline_rate + 0.10, (
            f"Hallucination rate REGRESSED: current={current_rate*100:.1f}% vs "
            f"baseline={baseline_rate*100:.1f}% (tolerance: 10%)"
        )
