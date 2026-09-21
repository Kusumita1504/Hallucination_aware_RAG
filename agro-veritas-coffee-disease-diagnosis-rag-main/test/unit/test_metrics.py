"""
Unit tests for all Phase 2 metric functions.
No LLM required — pure logic tests.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)


# ─── Re-implement metric functions here to keep tests self-contained ────────────

def normalize(text: str) -> str:
    return (text or "").strip().lower()


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
    exp_n = normalize(expected)
    return any(normalize(d) == exp_n for d, _ in candidates[:k])


def mrr(candidates: List[Tuple[str, float]], expected: str) -> float:
    exp_n = normalize(expected)
    for i, (d, _) in enumerate(candidates, 1):
        if normalize(d) == exp_n:
            return 1.0 / i
    return 0.0


def ml_precision(candidates, plausible, k):
    if not candidates:
        return 0.0
    pn = [normalize(d) for d in plausible]
    return sum(1 for d, _ in candidates[:k] if normalize(d) in pn) / min(k, len(candidates))


def ml_recall(candidates, plausible, k):
    if not plausible:
        return 1.0
    pn = [normalize(d) for d in plausible]
    pred_n = [normalize(d) for d, _ in candidates[:k]]
    return sum(1 for p in pn if p in pred_n) / len(plausible)


# ─── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_RESPONSE = """
CANDIDATE COMPARISON:
1. Coffee Leaf Rust: 85% - Orange powder on underside of leaves
2. Brown Eye Spot: 10% - No gray center observed
3. Phoma Leaf Spot: 5% - Unlikely given edge pattern

FINAL DIAGNOSIS:
DISEASE: Coffee Leaf Rust
CONFIDENCE: 85%
"""

THREE_CANDIDATES = [
    ("Coffee Leaf Rust", 0.85),
    ("Brown Eye Spot", 0.10),
    ("Phoma Leaf Spot", 0.05),
]


# ─── Candidate Extraction Tests ─────────────────────────────────────────────────

class TestCandidateExtraction:
    def test_extracts_correct_count(self):
        cands = extract_candidates(SAMPLE_RESPONSE)
        assert len(cands) == 3

    def test_first_candidate_is_correct(self):
        cands = extract_candidates(SAMPLE_RESPONSE)
        assert normalize(cands[0][0]) == "coffee leaf rust"

    def test_confidence_parsed_correctly(self):
        cands = extract_candidates(SAMPLE_RESPONSE)
        assert abs(cands[0][1] - 0.85) < 0.01
        assert abs(cands[1][1] - 0.10) < 0.01
        assert abs(cands[2][1] - 0.05) < 0.01

    def test_sorted_descending(self):
        cands = extract_candidates(SAMPLE_RESPONSE)
        confidences = [c for _, c in cands]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_input_returns_empty_list(self):
        assert extract_candidates("No candidate section.") == []

    def test_handles_decimal_percentages(self):
        text = """CANDIDATE COMPARISON:
1. Red Spider Mites: 87.5% - Bronze speckling
2. Nitrogen Deficiency: 12.5% - Possible
FINAL DIAGNOSIS:"""
        cands = extract_candidates(text)
        assert len(cands) == 2
        assert abs(cands[0][1] - 0.875) < 0.001

    def test_handles_primary_diagnosis_keyword(self):
        text = """CANDIDATE COMPARISON:
1. Anthracnose: 70% - Sunken lesions
2. Root Rot: 30% - Less likely
PRIMARY DIAGNOSIS:
DISEASE: Anthracnose"""
        cands = extract_candidates(text)
        assert len(cands) == 2

    def test_malformed_input_does_not_crash(self):
        malformed = "CANDIDATE COMPARISON:\nNo percentages here\nFINAL DIAGNOSIS:"
        cands = extract_candidates(malformed)
        assert isinstance(cands, list)


# ─── Top-K Accuracy Tests ───────────────────────────────────────────────────────

class TestTopKAccuracy:
    def test_top1_hit_rank1(self):
        assert top_k_hit(THREE_CANDIDATES, "Coffee Leaf Rust", k=1) is True

    def test_top1_miss_rank2(self):
        assert top_k_hit(THREE_CANDIDATES, "Brown Eye Spot", k=1) is False

    def test_top1_miss_rank3(self):
        assert top_k_hit(THREE_CANDIDATES, "Phoma Leaf Spot", k=1) is False

    def test_top3_hit_rank2(self):
        assert top_k_hit(THREE_CANDIDATES, "Brown Eye Spot", k=3) is True

    def test_top3_hit_rank3(self):
        assert top_k_hit(THREE_CANDIDATES, "Phoma Leaf Spot", k=3) is True

    def test_top3_miss_absent_disease(self):
        assert top_k_hit(THREE_CANDIDATES, "Anthracnose", k=3) is False

    def test_case_insensitive_match(self):
        assert top_k_hit(THREE_CANDIDATES, "coffee leaf rust", k=1) is True
        assert top_k_hit(THREE_CANDIDATES, "BROWN EYE SPOT", k=3) is True

    def test_empty_candidates_always_miss(self):
        assert top_k_hit([], "Coffee Leaf Rust", k=1) is False

    def test_k_larger_than_candidates(self):
        single = [("Root Rot / Wilt Disease", 0.9)]
        assert top_k_hit(single, "Root Rot / Wilt Disease", k=5) is True


# ─── MRR Tests ──────────────────────────────────────────────────────────────────

class TestMRR:
    def test_rank1_mrr_is_one(self):
        assert mrr(THREE_CANDIDATES, "Coffee Leaf Rust") == 1.0

    def test_rank2_mrr_is_half(self):
        assert abs(mrr(THREE_CANDIDATES, "Brown Eye Spot") - 0.5) < 0.001

    def test_rank3_mrr_is_one_third(self):
        assert abs(mrr(THREE_CANDIDATES, "Phoma Leaf Spot") - 1/3) < 0.001

    def test_absent_disease_mrr_is_zero(self):
        assert mrr(THREE_CANDIDATES, "Anthracnose") == 0.0

    def test_empty_candidates_mrr_is_zero(self):
        assert mrr([], "Coffee Leaf Rust") == 0.0

    def test_single_candidate_mrr(self):
        single = [("Root Rot / Wilt Disease", 0.95)]
        assert mrr(single, "Root Rot / Wilt Disease") == 1.0
        assert mrr(single, "Anthracnose") == 0.0


# ─── Multi-Label Precision Tests ────────────────────────────────────────────────

class TestMultiLabelPrecision:
    def setup_method(self):
        self.candidates = [
            ("Coffee Leaf Rust", 0.60),
            ("Brown Eye Spot", 0.30),
            ("Phoma Leaf Spot", 0.10),
        ]
        self.plausible = ["Coffee Leaf Rust", "Brown Eye Spot"]

    def test_p_at_2_all_plausible(self):
        p = ml_precision(self.candidates, self.plausible, k=2)
        assert p == 1.0

    def test_p_at_3_two_thirds_plausible(self):
        p = ml_precision(self.candidates, self.plausible, k=3)
        assert abs(p - 2/3) < 0.001

    def test_p_at_1_rank1_plausible(self):
        p = ml_precision(self.candidates, self.plausible, k=1)
        assert p == 1.0

    def test_p_at_k_no_plausible_overlap(self):
        p = ml_precision(self.candidates, ["Anthracnose", "Root Rot"], k=3)
        assert p == 0.0

    def test_p_at_k_empty_candidates(self):
        assert ml_precision([], self.plausible, k=3) == 0.0


# ─── Multi-Label Recall Tests ───────────────────────────────────────────────────

class TestMultiLabelRecall:
    def setup_method(self):
        self.candidates = [
            ("Coffee Leaf Rust", 0.60),
            ("Brown Eye Spot", 0.25),
            ("Red Spider Mites", 0.10),
            ("Anthracnose", 0.05),
        ]
        self.plausible = ["Coffee Leaf Rust", "Brown Eye Spot", "Anthracnose"]

    def test_recall_at_3_two_thirds(self):
        r = ml_recall(self.candidates, self.plausible, k=3)
        assert abs(r - 2/3) < 0.001

    def test_recall_at_4_all_found(self):
        r = ml_recall(self.candidates, self.plausible, k=4)
        assert r == 1.0

    def test_recall_at_1_one_third(self):
        r = ml_recall(self.candidates, self.plausible, k=1)
        assert abs(r - 1/3) < 0.001

    def test_recall_empty_plausible_is_one(self):
        assert ml_recall(self.candidates, [], k=3) == 1.0

    def test_recall_no_overlap(self):
        r = ml_recall(self.candidates, ["Zinc Deficiency", "Iron Deficiency"], k=4)
        assert r == 0.0
