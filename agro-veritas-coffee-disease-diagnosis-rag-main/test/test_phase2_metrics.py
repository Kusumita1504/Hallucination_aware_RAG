"""
Test Phase 2 metrics implementation without semantic model dependencies
"""

import sys
from pathlib import Path
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize(text: str) -> str:
    """Normalize text for comparison"""
    return (text or "").strip().lower()


def extract_candidates_from_response(response_text: str):
    """Extract disease candidates with confidence scores from diagnosis response"""
    candidates = []
    match = re.search(r'CANDIDATE COMPARISON:?(.*?)(?:PRIMARY|FINAL|DIAGNOSIS:|$)', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        candidate_section = match.group(1)
        for line in candidate_section.split('\n'):
            if not line.strip():
                continue
            pattern = r'(\d+)\.\s*([^:]+):\s*(\d+(?:\.\d+)?)\s*%'
            candidate_match = re.search(pattern, line)
            if candidate_match:
                disease_name = candidate_match.group(2).strip()
                confidence = float(candidate_match.group(3)) / 100.0
                candidates.append((disease_name, confidence))
    return candidates


def top_k_accuracy(predicted_candidates, expected_disease, k=1):
    """Check if expected disease is in top-k candidates"""
    expected_norm = normalize(expected_disease)
    for i, (disease, _) in enumerate(predicted_candidates[:k]):
        if normalize(disease) == expected_norm:
            return True
    return False


def mean_reciprocal_rank(predicted_candidates, expected_disease):
    """Calculate Mean Reciprocal Rank (MRR)"""
    expected_norm = normalize(expected_disease)
    for i, (disease, _) in enumerate(predicted_candidates, 1):
        if normalize(disease) == expected_norm:
            return 1.0 / i
    return 0.0


def multi_label_precision_at_k(predicted_candidates, plausible_diseases, k=3):
    """Calculate precision@k for ambiguous cases with multiple correct answers"""
    if not predicted_candidates:
        return 0.0
    plausible_norm = [normalize(d) for d in plausible_diseases]
    matches = 0
    for disease, _ in predicted_candidates[:k]:
        if normalize(disease) in plausible_norm:
            matches += 1
    return matches / min(k, len(predicted_candidates))


def multi_label_recall_at_k(predicted_candidates, plausible_diseases, k=3):
    """Calculate recall@k: what fraction of plausible diseases are in top-k?"""
    if not plausible_diseases:
        return 1.0
    predicted_norm = [normalize(d) for d, _ in predicted_candidates[:k]]
    plausible_norm = [normalize(d) for d in plausible_diseases]
    found = sum(1 for p in plausible_norm if p in predicted_norm)
    return found / len(plausible_diseases)


def test_candidate_extraction():
    """Test extracting candidates from response format"""
    response_text = """
    CANDIDATE COMPARISON:
    1. Coffee Leaf Rust: 85% - Orange powder on leaves
    2. Brown Eye Spot: 10% - But no gray center
    3. Phoma Leaf Spot: 5% - Unlikely
    
    PRIMARY DIAGNOSIS:
    Coffee Leaf Rust
    """
    
    candidates = extract_candidates_from_response(response_text)
    assert len(candidates) >= 2, f"Expected >= 2 candidates, got {len(candidates)}"
    assert normalize(candidates[0][0]) == normalize("Coffee Leaf Rust"), "First candidate should be Coffee Leaf Rust"
    assert 0.80 <= candidates[0][1] <= 0.90, f"Confidence should be ~0.85, got {candidates[0][1]}"
    print("✅ test_candidate_extraction passed")


def test_top_k_accuracy():
    """Test top-K accuracy metric"""
    candidates = [
        ("Coffee Leaf Rust", 0.85),
        ("Brown Eye Spot", 0.10),
        ("Phoma Leaf Spot", 0.05)
    ]
    
    # Top-1: Should find Coffee Leaf Rust
    assert top_k_accuracy(candidates, "Coffee Leaf Rust", k=1) == True
    assert top_k_accuracy(candidates, "Brown Eye Spot", k=1) == False
    
    # Top-3: Should find all three
    assert top_k_accuracy(candidates, "Coffee Leaf Rust", k=3) == True
    assert top_k_accuracy(candidates, "Brown Eye Spot", k=3) == True
    assert top_k_accuracy(candidates, "Phoma Leaf Spot", k=3) == True
    
    # Test case insensitivity
    assert top_k_accuracy(candidates, "coffee leaf rust", k=1) == True
    
    print("✅ test_top_k_accuracy passed")


def test_mean_reciprocal_rank():
    """Test MRR metric"""
    candidates = [
        ("Coffee Leaf Rust", 0.85),
        ("Brown Eye Spot", 0.10),
        ("Phoma Leaf Spot", 0.05)
    ]
    
    # If first is expected
    mrr1 = mean_reciprocal_rank(candidates, "Coffee Leaf Rust")
    assert mrr1 == 1.0, f"Expected 1.0, got {mrr1}"
    
    # If second is expected
    mrr2 = mean_reciprocal_rank(candidates, "Brown Eye Spot")
    assert mrr2 == 0.5, f"Expected 0.5, got {mrr2}"
    
    # If not found
    mrr_not_found = mean_reciprocal_rank(candidates, "Unknown Disease")
    assert mrr_not_found == 0.0, f"Expected 0.0, got {mrr_not_found}"
    
    print("✅ test_mean_reciprocal_rank passed")


def test_multi_label_precision():
    """Test multi-label precision@k"""
    candidates = [
        ("Coffee Leaf Rust", 0.85),
        ("Brown Eye Spot", 0.10),
        ("Phoma Leaf Spot", 0.05)
    ]
    
    plausible = ["Coffee Leaf Rust", "Brown Eye Spot"]
    
    # Top-2: Both are in plausible
    p_at_2 = multi_label_precision_at_k(candidates, plausible, k=2)
    assert p_at_2 == 1.0, f"Expected 1.0, got {p_at_2}"
    
    # Top-3: 2 out of 3 are plausible
    p_at_3 = multi_label_precision_at_k(candidates, plausible, k=3)
    assert p_at_3 == 2/3, f"Expected 0.667, got {p_at_3}"
    
    print("✅ test_multi_label_precision passed")


def test_multi_label_recall():
    """Test multi-label recall@k"""
    candidates = [
        ("Coffee Leaf Rust", 0.85),
        ("Brown Eye Spot", 0.10),
        ("Phoma Leaf Spot", 0.05),
        ("Red Spider Mites", 0.0)
    ]
    
    plausible = ["Coffee Leaf Rust", "Brown Eye Spot", "Red Spider Mites"]
    
    # Top-3: 2 out of 3 plausible diseases found
    r_at_3 = multi_label_recall_at_k(candidates, plausible, k=3)
    assert r_at_3 == 2/3, f"Expected 0.667, got {r_at_3}"
    
    # Top-4: All 3 plausible found
    r_at_4 = multi_label_recall_at_k(candidates, plausible, k=4)
    assert r_at_4 == 1.0, f"Expected 1.0, got {r_at_4}"
    
    print("✅ test_multi_label_recall passed")


def test_dataset_v2_format():
    """Validate evaluation_dataset_v2.json format"""
    dataset_path = PROJECT_ROOT / "test" / "evaluation_dataset_v2.json"
    
    if not dataset_path.exists():
        print(f"⚠️  Dataset not found at {dataset_path}")
        return
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    cases = data.get('cases', [])
    assert len(cases) >= 50, f"Expected >= 50 cases, got {len(cases)}"
    
    # Check case format
    for case in cases[:5]:
        assert 'id' in case, "Missing 'id' field"
        assert 'query' in case, "Missing 'query' field"
        assert 'expected_disease' in case, "Missing 'expected_disease' field"
        assert 'relevant_keywords' in case, "Missing 'relevant_keywords' field"
    
    # Count ambiguous cases
    ambiguous = [c for c in cases if c.get('disambiguation_required', False)]
    assert len(ambiguous) >= 8, f"Expected >= 8 ambiguous cases, got {len(ambiguous)}"
    
    print(f"✅ test_dataset_v2_format passed ({len(cases)} cases, {len(ambiguous)} ambiguous)")


if __name__ == "__main__":
    print("\nPhase 2: Testing Metrics Implementation\n")
    print("="*70)
    
    try:
        test_candidate_extraction()
        test_top_k_accuracy()
        test_mean_reciprocal_rank()
        test_multi_label_precision()
        test_multi_label_recall()
        test_dataset_v2_format()
        
        print("\n" + "="*70)
        print("✅ All Phase 2 metrics tests passed!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
