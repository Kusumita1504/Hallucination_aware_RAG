"""
Test Hybrid JSON + PDF Retrieval System
Tests balanced disease distribution
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Ensure both project root (for `config`) and src (for package imports) are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController

def test_hybrid_retrieval():
    """Test hybrid retrieval with diverse symptom inputs"""
    
    print("=" * 80)
    print("HYBRID RETRIEVAL SYSTEM TEST")
    print("Testing balanced disease distribution (JSON 60% + PDF 40%)")
    print("=" * 80)
    
    # Test cases with diverse symptoms
    test_cases = [
        "Yellow leaves on older branches",
        "Brown spots with orange powder on leaves",
        "Small holes in coffee berries",
        "Wilting plant despite watering",
        "Yellowing between leaf veins",
        "Leaf margins turning brown and crispy",
        "Small narrow new leaves bunched together",
        "Sudden branch dieback with sawdust",
    ]
    
    try:
        # Initialize controller
        print("\n[1/3] Initializing system...")
        controller = CoffeeDiagnosisController(
            data_dir=str(PROJECT_ROOT / "data" / "pdfs"),
            vector_db_path=str(PROJECT_ROOT / "data" / "vector_db")
        )
        
        print("\n[2/3] Running test queries...")
        print("-" * 80)
        
        results = {}
        
        for i, query in enumerate(test_cases, 1):
            print(f"\nTest {i}/{len(test_cases)}: {query}")
            
            # Start diagnosis
            response = controller.start_diagnosis(query)
            
            if response['status'] == 'question':
                # Continue the loop with a neutral uncertain answer
                # until diagnosis is produced or max turns reached.
                turns = 0
                current = response
                while current.get('status') == 'question' and turns < 5:
                    current = controller.submit_answer("not sure")
                    turns += 1

                if current.get('status') == 'diagnosis':
                    diagnosis = current['diagnosis']
                    disease = diagnosis.disease_name
                    print(f"  → Diagnosis: {disease}")

                    # Track disease counts
                    results[disease] = results.get(disease, 0) + 1

                    # Show sources
                    ctx = controller.state_manager.state.retrieved_context
                    json_count = sum(1 for c in ctx if c.get('source_type') == 'JSON')
                    pdf_count = sum(1 for c in ctx if c.get('source_type') == 'PDF')
                    print(f"  → Sources: {json_count} JSON, {pdf_count} PDF")
                else:
                    print("  → No diagnosis returned within expected turns")
            elif response['status'] == 'diagnosis':
                diagnosis = response['diagnosis']
                disease = diagnosis.disease_name
                print(f"  → Diagnosis: {disease}")
                results[disease] = results.get(disease, 0) + 1
                
                ctx = controller.state_manager.state.retrieved_context
                json_count = sum(1 for c in ctx if c.get('source_type') == 'JSON')
                pdf_count = sum(1 for c in ctx if c.get('source_type') == 'PDF')
                print(f"  → Sources: {json_count} JSON, {pdf_count} PDF")
            
            # Reset for next test
            controller.reset()
        
        print("\n" + "=" * 80)
        print("[3/3] RESULTS SUMMARY")
        print("=" * 80)
        
        print("\nDisease Distribution:")
        for disease, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(test_cases)) * 100
            print(f"  {disease}: {count} ({percentage:.1f}%)")
        
        # Check if Coffee Leaf Rust dominates
        clf_count = results.get("Coffee Leaf Rust", 0)
        if clf_count >= len(test_cases) * 0.7:
            print("\n❌ FAILED: Coffee Leaf Rust still dominates (≥70%)")
            print("   System is still biased!")
        else:
            print("\n✅ PASSED: No single disease dominates!")
            print(f"   Coffee Leaf Rust: {clf_count}/{len(test_cases)} ({(clf_count/len(test_cases)*100):.1f}%)")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_retrieval()
