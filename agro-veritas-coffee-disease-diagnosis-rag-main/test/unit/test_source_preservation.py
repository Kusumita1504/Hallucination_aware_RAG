"""Retrieval evaluation retains source citations for filtered evidence."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from coffee_diagnosis.rag.retrieval_evaluator import RetrievalEvaluator


def test_evaluation_preserves_json_source_type():
    evaluator = RetrievalEvaluator(relevance_threshold=0)
    evaluated = evaluator.evaluate_chunks(
        [
            {
                "content": "Coffee leaf rust causes orange spots on coffee leaves.",
                "source": "disease_knowledge.json",
                "source_type": "JSON",
                "metadata": {"disease_name": "Coffee Leaf Rust"},
            }
        ],
        "coffee orange spots leaves",
    )
    assert evaluated[0]["source_type"] == "JSON"
