"""
Test script for multi-turn clarification questions
Tests the new start_diagnosis() and submit_answer() methods
"""

import sys
from pathlib import Path

# Ensure both project root (for `config`) and src (for package imports) are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController

def test_multiturn():
    """Test multi-turn diagnosis process"""
    print("[TEST] Initializing Coffee Disease Diagnosis System...")
    controller = CoffeeDiagnosisController(
        data_dir=str(PROJECT_ROOT / "data" / "pdfs"),
        vector_db_path=str(PROJECT_ROOT / "data" / "vector_db")
    )

    # Test 1: Simple symptom that should trigger questions
    print("\n" + "="*60)
    print("TEST 1: Simple symptom query")
    print("="*60)
    initial_query = "Leaves are turning yellow"
    print(f"Initial Query: {initial_query}")

    result = controller.start_diagnosis(initial_query)
    print(f"\nStatus: {result['status']}")

    if result['status'] == 'question':
        print(f"Question Asked: {result['question']}")
        print("\nExpected: System should ask clarification questions")

        # Simulate user answering
        print("\n" + "-"*60)
        user_answer = "The yellow color is with brown spots on the edges"
        print(f"User Answer: {user_answer}")

        result = controller.submit_answer(user_answer)
        print(f"\nStatus: {result['status']}")

        if result['status'] == 'question':
            print(f"Next Question: {result['question']}")
        elif result['status'] == 'diagnosis':
            print(f"Diagnosis: {result['diagnosis'].disease_name}")
            print(f"Confidence: {result['diagnosis'].confidence * 100:.1f}%")

    elif result['status'] == 'diagnosis':
        print("Diagnosis generated directly (no questions needed)")
        print(f"Disease: {result['diagnosis'].disease_name}")

    # Test 2: Another simple query
    print("\n" + "="*60)
    print("TEST 2: Another symptom query")
    print("="*60)
    controller.reset()

    initial_query2 = "Stem is rotting"
    print(f"Initial Query: {initial_query2}")

    result = controller.start_diagnosis(initial_query2)
    print(f"\nStatus: {result['status']}")

    if result['status'] == 'question':
        print(f"Question Asked: {result['question']}")
    else:
        print(f"Diagnosis: {result['diagnosis'].disease_name}")

    print("\n" + "="*60)
    print("Tests completed successfully!")
    print("="*60)

if __name__ == "__main__":
    test_multiturn()
