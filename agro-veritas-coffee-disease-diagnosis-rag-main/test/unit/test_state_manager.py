"""
Unit tests for StateManager.
No LLM required — pure state logic tests.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from coffee_diagnosis.rag.state_manager import StateManager
from config import settings


class TestStateManagerInit:
    def test_default_max_questions_uses_settings(self):
        sm = StateManager()
        assert sm.max_questions == settings.MAX_QUESTIONS

    def test_custom_max_questions_overrides_settings(self):
        sm = StateManager(max_questions=3)
        assert sm.max_questions == 3

    def test_custom_max_questions_7(self):
        sm = StateManager(max_questions=7)
        assert sm.max_questions == 7

    def test_initial_state_is_empty(self):
        sm = StateManager()
        assert sm.state.initial_query == ""
        assert sm.state.questions_asked == 0
        assert sm.state.user_responses == []


class TestStateManagerInitialize:
    def test_initialize_sets_query(self):
        sm = StateManager()
        sm.initialize("My coffee leaves have orange spots")
        assert sm.state.initial_query == "My coffee leaves have orange spots"

    def test_initialize_sets_low_confidence(self):
        sm = StateManager()
        sm.initialize("test")
        assert sm.state.confidence == 0.2

    def test_initialize_resets_previous_state(self):
        sm = StateManager()
        sm.initialize("query 1")
        sm.add_user_response("response 1")
        sm.initialize("query 2")
        assert sm.state.user_responses == []
        assert sm.state.initial_query == "query 2"


class TestConfidenceTracking:
    def test_confidence_increases_on_user_response(self):
        sm = StateManager()
        sm.initialize("test")
        initial = sm.confidence
        sm.add_user_response("leaves are yellow")
        assert sm.confidence == initial + 0.15

    def test_confidence_capped_at_1(self):
        sm = StateManager()
        sm.initialize("test")
        for _ in range(20):
            sm.add_user_response("answer")
        assert sm.confidence <= 1.0

    def test_set_confidence_directly(self):
        sm = StateManager()
        sm.initialize("test")
        sm.set_confidence(0.9)
        assert sm.confidence == 0.9

    def test_set_confidence_clamps_to_zero(self):
        sm = StateManager()
        sm.set_confidence(-0.5)
        assert sm.confidence == 0.0

    def test_set_confidence_clamps_to_one(self):
        sm = StateManager()
        sm.set_confidence(1.5)
        assert sm.confidence == 1.0


class TestShouldStop:
    """Tests the stop condition logic."""

    def setup_method(self):
        """Create a fresh StateManager with max_questions=5 before each test."""
        self.sm = StateManager(max_questions=5)
        self.sm.initialize("test query")
        # Add some ambiguity so it doesn't stop for "no missing info"
        self.sm.update_detected_ambiguities({"missing": {"color": "unknown"}})

    def test_stops_at_hard_cap(self):
        self.sm.state.questions_asked = 5
        assert self.sm.should_stop() is True

    def test_does_not_stop_below_cap(self):
        self.sm.state.questions_asked = 4
        self.sm.state.confidence = 0.5
        assert self.sm.should_stop() is False

    def test_stops_at_high_confidence(self):
        self.sm.state.questions_asked = 2
        self.sm.state.confidence = 0.85
        assert self.sm.should_stop() is True

    def test_does_not_stop_at_moderate_confidence(self):
        self.sm.state.questions_asked = 2
        self.sm.state.confidence = 0.7
        assert self.sm.should_stop() is False

    def test_stops_when_single_disease_remains(self):
        self.sm.state.questions_asked = 2
        self.sm.state.confidence = 0.5
        self.sm.set_possible_diseases(["Coffee Leaf Rust"])
        assert self.sm.should_stop() is True

    def test_does_not_stop_with_multiple_diseases(self):
        self.sm.state.questions_asked = 2
        self.sm.state.confidence = 0.5
        self.sm.set_possible_diseases(["Coffee Leaf Rust", "Brown Eye Spot"])
        assert self.sm.should_stop() is False

    def test_stops_when_no_missing_info(self):
        self.sm.state.questions_asked = 2
        self.sm.state.confidence = 0.5
        self.sm.update_detected_ambiguities({"missing": {}})  # No missing info
        assert self.sm.should_stop() is True

    def test_hard_cap_3_respected(self):
        sm3 = StateManager(max_questions=3)
        sm3.initialize("test")
        sm3.update_detected_ambiguities({"missing": {"color": "unknown"}})
        sm3.state.questions_asked = 3
        sm3.state.confidence = 0.5
        assert sm3.should_stop() is True

    def test_hard_cap_3_not_yet_reached(self):
        sm3 = StateManager(max_questions=3)
        sm3.initialize("test")
        sm3.update_detected_ambiguities({"missing": {"color": "unknown"}})
        sm3.state.questions_asked = 2
        sm3.state.confidence = 0.5
        assert sm3.should_stop() is False


class TestQuestionsTracking:
    def test_add_system_question_increments_count(self):
        sm = StateManager()
        sm.initialize("test")
        assert sm.questions_asked == 0
        sm.add_system_question("What color are the spots?", attribute="color")
        assert sm.questions_asked == 1

    def test_attribute_tracking(self):
        sm = StateManager()
        sm.initialize("test")
        sm.add_system_question("Q1?", attribute="color")
        sm.add_system_question("Q2?", attribute="location")
        assert "color" in sm.state.asked_about_attributes
        assert "location" in sm.state.asked_about_attributes

    def test_duplicate_attribute_not_added_twice(self):
        sm = StateManager()
        sm.initialize("test")
        sm.add_system_question("Q1?", attribute="color")
        sm.add_system_question("Q2?", attribute="color")
        assert sm.state.asked_about_attributes.count("color") == 1


class TestStateSummary:
    def test_summary_contains_required_keys(self):
        sm = StateManager()
        sm.initialize("test")
        summary = sm.get_state_summary()
        required_keys = [
            "initial_query", "confidence", "questions_asked",
            "max_questions", "ambiguities", "possible_diseases", "history_length"
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_summary_max_questions_reflects_setting(self):
        sm = StateManager(max_questions=7)
        sm.initialize("test")
        assert sm.get_state_summary()["max_questions"] == 7


class TestReset:
    def test_reset_clears_state(self):
        sm = StateManager()
        sm.initialize("test query")
        sm.add_user_response("response")
        sm.add_system_question("question", attribute="color")
        sm.reset()
        assert sm.state.initial_query == ""
        assert sm.state.questions_asked == 0
        assert sm.state.user_responses == []

    def test_max_questions_preserved_after_reset(self):
        sm = StateManager(max_questions=4)
        sm.initialize("test")
        sm.reset()
        assert sm.max_questions == 4  # Override preserved
