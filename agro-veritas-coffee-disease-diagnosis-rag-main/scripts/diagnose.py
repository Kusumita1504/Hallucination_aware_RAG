"""
Diagnostic script to test the system step by step
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("\n" + "="*60)
print("COFFEE DIAGNOSIS SYSTEM - DIAGNOSTIC TEST")
print("="*60 + "\n")

try:
    print("[1/5] Testing PDF Loader...")
    from coffee_diagnosis.core.pdf_loader import PDFLoader
    loader = PDFLoader(data_dir=str(Path(__file__).parent.parent / "data" / "pdfs"))
    docs = loader.load_pdfs()
    print(f"  OK - Loaded {len(docs)} document chunks\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

try:
    print("[2/5] Testing Vector Store...")
    from coffee_diagnosis.core.vector_store import FAISSVectorStore
    vector_store = FAISSVectorStore(vector_db_path=str(Path(__file__).parent.parent / "data" / "vector_db"))
    vector_store.create_index(docs)
    print(f"  OK - FAISS index created\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

try:
    print("[3/5] Testing Retriever...")
    from coffee_diagnosis.core.retriever import Retriever
    retriever = Retriever(vector_store)
    results = retriever.retrieve("yellow leaves")
    print(f"  OK - Retrieved {len(results)} documents\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

try:
    print("[4/5] Testing Ambiguity Detector...")
    from coffee_diagnosis.rag.ambiguity_detector import AmbiguityDetector
    detector = AmbiguityDetector()
    ambiguity = detector.detect_ambiguity("Leaves are turning yellow")
    print(f"  OK - Detected ambiguities: {list(ambiguity['missing'].keys())}\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

try:
    print("[5/5] Testing Diagnosis Generator...")
    from coffee_diagnosis.diagnosis.diagnosis_generator import DiagnosisGenerator
    gen = DiagnosisGenerator()
    print(f"  OK - Diagnosis generator ready\n")
except Exception as e:
    print(f"  FAILED: {e}\n")
    sys.exit(1)

print("="*60)
print("ALL TESTS PASSED - System is ready!")
print("="*60)
print("\nRun: streamlit run ui/streamlit_app.py")
print("Then open: http://localhost:8501\n")
