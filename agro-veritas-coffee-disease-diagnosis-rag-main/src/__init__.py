"""
Coffee Disease Diagnosis System
A comprehensive AI system for diagnosing coffee diseases in Karnataka
"""

__version__ = "1.0.0"
__author__ = "AI Agriculture Expert"

from .controller import CoffeeDiagnosisController
from .state_manager import StateManager
from .diagnosis_generator import Diagnosis

__all__ = [
    'CoffeeDiagnosisController',
    'StateManager',
    'Diagnosis'
]
