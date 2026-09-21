"""
Integration tests for the full CoffeeDiagnosisController pipeline.
Requires a running local LLM (Ollama) — marked as @pytest.mark.slow.

Run with:
    pytest test/integration/ -v -m slow
    pytest test/integration/ -v --timeout=120  # with 2-min timeout per test
"""

import sys
from pathlib import Path
from typing import Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# ─── Single-Turn Tests ──────────────────────────────────────────────────────────

@pytest.mark.timeout(120)
def test_start_diagnosis_returns_dict(fresh_controller):
    """start_diagnosis() must always return a dict with 'status' key."""
    result = fresh_controller.start_diagnosis("yellow spots on coffee leaves")
    assert isinstance(result, dict), "Result must be a dict"
    assert "status" in result, "Result must have 'status' key"
    assert result["status"] in ("question", "diagnosis", "off_topic"), (
        f"Unknown status: {result['status']}"
    )


@pytest.mark.timeout(120)
def test_off_topic_query_rejected(fresh_controller):
    """Non-coffee queries should return off_topic status."""
    result = fresh_controller.start_diagnosis("What is the capital of France?")
    assert result["status"] == "off_topic", (
        f"Off-topic query should return 'off_topic', got '{result['status']}'"
    )
    assert "message" in result, "off_topic response must have 'message' key"


@pytest.mark.timeout(120)
def test_coffee_query_accepted(fresh_controller):
    """A valid coffee symptom query should not be rejected as off-topic."""
    result = fresh_controller.start_diagnosis(
        "My coffee plant leaves have orange powder on the underside"
    )
    assert result["status"] in ("question", "diagnosis"), (
        f"Valid coffee query should not be off_topic, got '{result['status']}'"
    )


# ─── Multi-Turn Tests ───────────────────────────────────────────────────────────

@pytest.mark.timeout(180)
def test_multi_turn_reaches_diagnosis(fresh_controller):
    """The pipeline should eventually return a diagnosis status."""
    result = fresh_controller.start_diagnosis(
        "Yellow-orange powder on underside of coffee leaves"
    )

    loops = 0
    while result.get("status") == "question" and loops < 10:
        result = fresh_controller.submit_answer("yes, the powder is orange and chalky")
        loops += 1

    assert result["status"] == "diagnosis", (
        f"Expected 'diagnosis' after {loops} turns, got '{result['status']}'"
    )


@pytest.mark.timeout(180)
def test_diagnosis_result_has_required_fields(fresh_controller):
    """Diagnosis result must have all expected fields."""
    result = fresh_controller.start_diagnosis(
        "Coffee leaves showing rust-colored powdery spots"
    )

    loops = 0
    while result.get("status") == "question" and loops < 10:
        result = fresh_controller.submit_answer("not sure")
        loops += 1

    assert result["status"] == "diagnosis"

    # Check top-level fields
    assert "diagnosis" in result, "Missing 'diagnosis' key"
    assert "verification" in result, "Missing 'verification' key"
    assert "state_summary" in result, "Missing 'state_summary' key"

    # Check diagnosis object fields
    diag = result["diagnosis"]
    assert hasattr(diag, "disease_name"), "Diagnosis missing disease_name"
    assert hasattr(diag, "confidence"), "Diagnosis missing confidence"
    assert hasattr(diag, "reason"), "Diagnosis missing reason"
    assert hasattr(diag, "treatment"), "Diagnosis missing treatment"

    # Check verification fields
    v = result["verification"]
    assert "consistent" in v, "Verification missing 'consistent' key"
    assert "consistency_score" in v, "Verification missing 'consistency_score' key"
    assert "hallucination_detected" in v, "Verification missing 'hallucination_detected' key"


@pytest.mark.timeout(60)
def test_hallucination_check_produces_valid_score(fresh_controller):
    """Hallucination checker consistency_score must be between 0 and 1."""
    result = fresh_controller.start_diagnosis("brown spots with yellow halos on leaves")

    loops = 0
    while result.get("status") == "question" and loops < 10:
        result = fresh_controller.submit_answer("not sure")
        loops += 1

    if result.get("status") == "diagnosis":
        v = result.get("verification", {})
        score = v.get("consistency_score", -1)
        assert 0.0 <= score <= 1.0, (
            f"consistency_score must be in [0, 1], got {score}"
        )


@pytest.mark.timeout(60)
def test_state_summary_max_questions_is_5(fresh_controller):
    """State summary should reflect the new MAX_QUESTIONS=5 setting."""
    fresh_controller.start_diagnosis("yellow leaves with no spots")

    summary = fresh_controller.state_manager.get_state_summary()
    assert summary.get("max_questions") == 5, (
        f"max_questions in state_summary should be 5, got {summary.get('max_questions')}"
    )


@pytest.mark.timeout(60)
def test_reset_clears_state_for_new_session(fresh_controller):
    """After reset(), a new diagnosis should start fresh."""
    fresh_controller.start_diagnosis("first query about leaves")
    fresh_controller.reset()

    result = fresh_controller.start_diagnosis("second query about stems")
    # Should not carry over state from first query
    assert fresh_controller.state_manager.state.initial_query == "second query about stems"
    assert fresh_controller.state_manager.state.questions_asked == 0 or (
        result["status"] in ("question", "diagnosis")  # Pipeline ran correctly
    )


# ─── Disease-Specific Tests ─────────────────────────────────────────────────────

@pytest.mark.timeout(180)
def test_coffee_leaf_rust_query(fresh_controller):
    """
    Coffee Leaf Rust is the most common disease.
    System should produce a plausible diagnosis.
    """
    result = fresh_controller.start_diagnosis(
        "Orange powder on underside of leaves with yellow halos"
    )

    loops = 0
    while result.get("status") == "question" and loops < 10:
        answers = [
            "yes orange chalky powder",
            "on the underside of leaves",
            "leaves are yellowing and falling",
            "not sure",
        ]
        result = fresh_controller.submit_answer(
            answers[loops] if loops < len(answers) else "not sure"
        )
        loops += 1

    assert result["status"] == "diagnosis"
    diag = result["diagnosis"]

    # Plausible diagnoses for this symptom set
    plausible = ["coffee leaf rust", "brown eye spot", "phoma leaf spot"]
    predicted = (diag.disease_name or "").strip().lower()
    is_plausible = any(p in predicted for p in plausible)

    assert is_plausible, (
        f"Unexpected diagnosis '{diag.disease_name}' for classic leaf rust symptoms. "
        f"Expected one of: Coffee Leaf Rust, Brown Eye Spot, Phoma Leaf Spot"
    )
