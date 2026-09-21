"""
Unit tests for multi-label evaluation of ambiguous disease cases.
Tests specifically target the 10 ambiguous cases in evaluation_dataset_v2.json.
No LLM required.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def ml_precision(candidates, plausible, k=3):
    if not candidates:
        return 0.0
    pn = [normalize(d) for d in plausible]
    return sum(1 for d, _ in candidates[:k] if normalize(d) in pn) / min(k, len(candidates))


def ml_recall(candidates, plausible, k=3):
    if not plausible:
        return 1.0
    pn = [normalize(d) for d in plausible]
    pred_n = [normalize(d) for d, _ in candidates[:k]]
    return sum(1 for p in pn if p in pred_n) / len(plausible)


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── Dataset fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ambiguous_cases() -> List[Dict]:
    path = PROJECT_ROOT / "test" / "evaluation_dataset_v2.json"
    if not path.exists():
        pytest.skip(f"evaluation_dataset_v2.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = [c for c in data.get("cases", []) if c.get("disambiguation_required", False)]
    if not cases:
        pytest.skip("No ambiguous cases found in dataset")
    return cases


# ─── Dataset structure tests ────────────────────────────────────────────────────

class TestAmbiguousCasesDataset:
    def test_at_least_8_ambiguous_cases(self, ambiguous_cases):
        assert len(ambiguous_cases) >= 8, (
            f"Expected ≥ 8 ambiguous cases, got {len(ambiguous_cases)}"
        )

    def test_all_have_plausible_alternatives(self, ambiguous_cases):
        for case in ambiguous_cases:
            cid = case.get("id", "?")
            alts = case.get("plausible_alternatives", [])
            assert len(alts) >= 1, (
                f"Case {cid}: disambiguation_required=True but no plausible_alternatives"
            )

    def test_all_have_expected_disease(self, ambiguous_cases):
        for case in ambiguous_cases:
            assert case.get("expected_disease"), (
                f"Case {case.get('id', '?')} missing expected_disease"
            )

    def test_primary_disease_not_in_alternatives(self, ambiguous_cases):
        """Primary disease should not also appear in plausible_alternatives."""
        for case in ambiguous_cases:
            expected = normalize(case.get("expected_disease", ""))
            alts = [normalize(a) for a in case.get("plausible_alternatives", [])]
            assert expected not in alts, (
                f"Case {case.get('id', '?')}: expected_disease is also in plausible_alternatives"
            )

    def test_covers_multiple_disambiguation_scenarios(self, ambiguous_cases):
        """Check that different types of ambiguities are covered."""
        queries = [c.get("query", "").lower() for c in ambiguous_cases]
        query_text = " ".join(queries)
        # Should cover at least some of: rust, eye spot, deficiency, borer, wilt
        keywords_covered = [kw for kw in ["rust", "spot", "deficiency", "borer", "wilt"]
                            if kw in query_text]
        assert len(keywords_covered) >= 2, (
            f"Ambiguous cases only cover: {keywords_covered}. Need more diverse scenarios."
        )


# ─── Multi-label metric tests with realistic scenarios ──────────────────────────

class TestRustVsEyeSpotDisambiguation:
    """Coffee Leaf Rust vs Brown Eye Spot is the most common ambiguous pair."""

    def setup_method(self):
        # Simulated candidate list for a rust/eye-spot ambiguous query
        self.candidates_rust_correct = [
            ("Coffee Leaf Rust", 0.65),
            ("Brown Eye Spot", 0.30),
            ("Phoma Leaf Spot", 0.05),
        ]
        self.candidates_eyespot_correct = [
            ("Brown Eye Spot", 0.55),
            ("Coffee Leaf Rust", 0.40),
            ("Phoma Leaf Spot", 0.05),
        ]
        self.plausible = ["Coffee Leaf Rust", "Brown Eye Spot"]

    def test_precision_when_both_in_top2(self):
        p = ml_precision(self.candidates_rust_correct, self.plausible, k=2)
        assert p == 1.0  # Both top-2 are plausible

    def test_recall_when_both_in_top2(self):
        r = ml_recall(self.candidates_rust_correct, self.plausible, k=2)
        assert r == 1.0  # Both plausible diseases found in top-2

    def test_f1_perfect_when_both_ranked_high(self):
        p = ml_precision(self.candidates_rust_correct, self.plausible, k=2)
        r = ml_recall(self.candidates_rust_correct, self.plausible, k=2)
        assert f1_score(p, r) == 1.0

    def test_precision_drops_when_irrelevant_ranked_high(self):
        bad_candidates = [
            ("Anthracnose", 0.50),  # Not plausible
            ("Coffee Leaf Rust", 0.40),
            ("Brown Eye Spot", 0.10),
        ]
        p = ml_precision(bad_candidates, self.plausible, k=2)
        assert p == 0.5  # 1/2 top-2 candidates are plausible


class TestDeficiencyDisambiguation:
    """Nutrient deficiency cases are often ambiguous (N vs Fe vs Mg)."""

    def setup_method(self):
        self.plausible = ["Nitrogen Deficiency", "Iron Deficiency", "Magnesium Deficiency"]

    def test_recall_at_3_when_all_three_ranked(self):
        candidates = [
            ("Nitrogen Deficiency", 0.50),
            ("Iron Deficiency", 0.30),
            ("Magnesium Deficiency", 0.20),
        ]
        r = ml_recall(candidates, self.plausible, k=3)
        assert r == 1.0

    def test_recall_at_2_is_two_thirds(self):
        candidates = [
            ("Nitrogen Deficiency", 0.50),
            ("Iron Deficiency", 0.30),
            ("Potassium Deficiency", 0.20),  # Not in plausible set
        ]
        r = ml_recall(candidates, self.plausible, k=2)
        assert abs(r - 2/3) < 0.001

    def test_precision_penalizes_wrong_deficiency(self):
        candidates = [
            ("Potassium Deficiency", 0.60),  # Not in plausible set
            ("Zinc Deficiency", 0.30),       # Not in plausible set
            ("Nitrogen Deficiency", 0.10),
        ]
        p = ml_precision(candidates, self.plausible, k=3)
        assert abs(p - 1/3) < 0.001


class TestMultiLabelF1:
    """Tests F1 score as a combined metric for ambiguous cases."""

    def test_perfect_f1(self):
        assert f1_score(1.0, 1.0) == 1.0

    def test_zero_f1_when_no_predictions(self):
        assert f1_score(0.0, 0.0) == 0.0

    def test_f1_harmonic_mean(self):
        p, r = 0.8, 0.6
        expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)
        assert abs(f1_score(p, r) - expected) < 0.001

    def test_f1_penalizes_precision_recall_imbalance(self):
        balanced = f1_score(0.7, 0.7)
        imbalanced = f1_score(1.0, 0.4)
        assert balanced > imbalanced


class TestDatasetAmbiguousScoringSimulation:
    """Simulate scoring all ambiguous cases as if the system got them right."""

    def test_all_correct_gives_perfect_scores(self, ambiguous_cases):
        """If system always ranks expected disease first, all metrics should be 1.0."""
        precisions = []
        recalls = []

        for case in ambiguous_cases:
            expected = case["expected_disease"]
            plausible = [expected] + case.get("plausible_alternatives", [])

            # Perfect system: expected disease ranked 1st with high confidence
            perfect_candidates = [(expected, 0.90)] + [
                (alt, 0.05) for alt in case.get("plausible_alternatives", [])[:2]
            ]

            p = ml_precision(perfect_candidates, plausible, k=3)
            r = ml_recall(perfect_candidates, plausible, k=3)  # k=3 consistent with precision
            precisions.append(p)
            recalls.append(r)

        avg_p = sum(precisions) / len(precisions)
        avg_r = sum(recalls) / len(recalls)

        assert avg_p >= 0.9, f"Perfect system should have precision ≥ 0.9, got {avg_p:.3f}"
        assert avg_r >= 0.9, f"Perfect system should have recall ≥ 0.9, got {avg_r:.3f}"

    def test_random_system_gets_poor_scores(self, ambiguous_cases):
        """A system that always predicts wrong diseases should score near 0."""
        precisions = []

        wrong_diseases = ["Anthracnose", "White Stem Borer", "Leaf Miner Damage"]

        for case in ambiguous_cases:
            expected = case["expected_disease"]
            plausible = [expected] + case.get("plausible_alternatives", [])

            # Wrong system: predicts unrelated diseases
            wrong_candidates = [(d, 0.9 - i * 0.2) for i, d in enumerate(wrong_diseases)]
            p = ml_precision(wrong_candidates, plausible, k=3)
            precisions.append(p)

        avg_p = sum(precisions) / len(precisions)
        assert avg_p < 0.3, f"Wrong system should have low precision, got {avg_p:.3f}"
