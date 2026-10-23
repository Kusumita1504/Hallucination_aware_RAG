"""
Coffee Disease Diagnosis Package
Main entry point with core exports
"""

__version__ = "1.0.0"
__all__ = ["CoffeeDiagnosisController", "StateManager", "Diagnosis"]


def __getattr__(name):
    """Load pipeline exports lazily so lightweight services need not import ML dependencies."""
    if name == "CoffeeDiagnosisController":
        from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController
        return CoffeeDiagnosisController
    if name == "StateManager":
        from coffee_diagnosis.rag.state_manager import StateManager
        return StateManager
    if name == "Diagnosis":
        from coffee_diagnosis.diagnosis.diagnosis_generator import Diagnosis
        return Diagnosis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
