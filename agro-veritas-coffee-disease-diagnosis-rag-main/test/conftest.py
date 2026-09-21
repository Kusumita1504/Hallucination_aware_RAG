"""
pytest configuration and shared fixtures for the coffee diagnosis test suite.
Fixtures are session-scoped where possible to avoid re-initializing the LLM/vector store.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)


# ─── Session-scoped fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def dataset_v1(project_root: Path) -> List[Dict]:
    """Load original 12-case evaluation dataset."""
    path = project_root / "test" / "evaluation_dataset.json"
    if not path.exists():
        pytest.skip(f"Dataset v1 not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


@pytest.fixture(scope="session")
def dataset_v2(project_root: Path) -> List[Dict]:
    """Load expanded 55-case evaluation dataset."""
    path = project_root / "test" / "evaluation_dataset_v2.json"
    if not path.exists():
        pytest.skip(f"Dataset v2 not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


@pytest.fixture(scope="session")
def controller(project_root: Path):
    """
    Initialize the CoffeeDiagnosisController once per test session.
    Marked slow — only used in integration tests.
    Requires local LLM (Ollama) to be running.
    """
    try:
        from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController
        ctrl = CoffeeDiagnosisController(
            data_dir=str(project_root / "data" / "pdfs"),
            vector_db_path=str(project_root / "data" / "vector_db")
        )
        return ctrl
    except Exception as e:
        pytest.skip(f"Controller init failed (LLM not available?): {e}")


@pytest.fixture(scope="function")
def fresh_controller(project_root: Path):
    """
    Fresh controller with reset state for each test function.
    Use for tests that need a clean state between calls.
    """
    try:
        from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController
        ctrl = CoffeeDiagnosisController(
            data_dir=str(project_root / "data" / "pdfs"),
            vector_db_path=str(project_root / "data" / "vector_db")
        )
        return ctrl
    except Exception as e:
        pytest.skip(f"Controller init failed (LLM not available?): {e}")


@pytest.fixture(scope="session")
def state_manager():
    """Fresh StateManager for unit testing."""
    from coffee_diagnosis.rag.state_manager import StateManager
    return StateManager()


@pytest.fixture(scope="session")
def settings():
    """Project settings."""
    from config import settings as s
    return s


# ─── pytest markers ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests that require a running LLM (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests (require full pipeline)"
    )
    config.addinivalue_line(
        "markers", "regression: marks regression baseline tests"
    )


# ─── __init__.py stubs for test subdirectories ──────────────────────────────────
# (Created programmatically to avoid manual file creation)

def pytest_sessionstart(session):
    """Create __init__.py in test subdirectories so pytest can discover them."""
    for subdir in ["unit", "integration", "regression"]:
        init_path = PROJECT_ROOT / "test" / subdir / "__init__.py"
        if not init_path.exists():
            init_path.touch()
