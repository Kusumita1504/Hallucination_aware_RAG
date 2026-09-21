"""
Coffee Disease Diagnosis Package
Main entry point with core exports
"""

from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController
from coffee_diagnosis.rag.state_manager import StateManager
from coffee_diagnosis.diagnosis.diagnosis_generator import Diagnosis

__version__ = "1.0.0"
__all__ = ["CoffeeDiagnosisController", "StateManager", "Diagnosis"]
