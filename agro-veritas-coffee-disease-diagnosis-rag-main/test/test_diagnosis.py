"""
CLI Test Script
Tests the coffee disease diagnosis system from command line
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController


def test_diagnosis(query: str):
    """Test diagnosis with a query"""
    print("\n" + "="*70)
    print("COFFEE DISEASE DIAGNOSIS SYSTEM - TEST MODE")
    print("="*70)

    try:
        # Initialize system
        controller = CoffeeDiagnosisController(
            data_dir=str(Path(__file__).parent.parent / "data" / "pdfs"),
            vector_db_path=str(Path(__file__).parent.parent / "data" / "vector_db")
        )

        # Run diagnosis
        print(f"\nTesting with query: '{query}'\n")
        results = controller.diagnose_web(query)

        # Print results
        controller.print_results(results)

        # Print conversation summary
        print("\n" + "="*70)
        print("CONVERSATION SUMMARY")
        print("="*70)
        print(f"\nInitial Query: {results['state_summary']['initial_query']}")
        print(f"Questions Asked: {results['state_summary']['questions_asked']}")
        print(f"Final Confidence: {results['state_summary']['confidence'] * 100:.1f}%")
        print(f"Ambiguities Detected: {len(results['state_summary']['ambiguities'].get('missing', {}))}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Run tests"""

    test_queries = [
        "Leaves are turning yellow with brown spots",
        "I see white powder on my coffee leaves",
        "The stem appears to be rotting from the base"
    ]

    print("\n🔬 Starting Coffee Disease Diagnosis System Tests...\n")

    # Test with first query only for demo
    test_diagnosis(test_queries[0])

    print("\n✅ Test completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom query passed as argument
        query = " ".join(sys.argv[1:])
        test_diagnosis(query)
    else:
        main()
