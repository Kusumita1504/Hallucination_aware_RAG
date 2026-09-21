"""
Unified test runner for all phases.
Wraps pytest with sensible defaults for each test level.

Usage:
    python test/run_tests.py --unit          Fast unit tests only (no LLM, ~5s)
    python test/run_tests.py --integration   Full pipeline tests (needs LLM, ~10min)
    python test/run_tests.py --regression    Baseline comparison (needs LLM + baseline)
    python test/run_tests.py --validate      Run validate_phase2.py (Phase 2 checks)
    python test/run_tests.py --all           Everything (unit + integration + regression)
    python test/run_tests.py                 Defaults to --unit
"""

import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEPARATOR = "=" * 72


def run(cmd: list, description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{SEPARATOR}")
    print(f"🔷 {description}")
    print(f"{SEPARATOR}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_unit_tests() -> int:
    return run(
        [
            sys.executable, "-m", "pytest",
            "test/unit/",
            "-v",
            "--tb=short",
            "-m", "not slow and not integration and not regression",
            "--timeout=30",
        ],
        "UNIT TESTS (no LLM required)"
    )


def run_integration_tests() -> int:
    return run(
        [
            sys.executable, "-m", "pytest",
            "test/integration/",
            "-v",
            "--tb=short",
            "-m", "slow or integration",
            "--timeout=300",
        ],
        "INTEGRATION TESTS (requires running LLM)"
    )


def run_regression_tests() -> int:
    return run(
        [
            sys.executable, "-m", "pytest",
            "test/regression/",
            "-v",
            "--tb=short",
            "-m", "regression",
            "--timeout=600",
        ],
        "REGRESSION TESTS (baseline comparison)"
    )


def run_phase2_validation() -> int:
    return run(
        [sys.executable, "test/validate_phase2.py"],
        "PHASE 2 VALIDATION (metric function checks, no LLM)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified test runner for coffee diagnosis RAG project"
    )
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests (needs LLM)")
    parser.add_argument("--regression", action="store_true", help="Run regression tests (needs baseline)")
    parser.add_argument("--validate", action="store_true", help="Run Phase 2 metric validation")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    # Default to unit tests
    if not any([args.unit, args.integration, args.regression, args.validate, args.all]):
        args.unit = True

    total_failures = 0

    print(f"\n{'='*72}")
    print("COFFEE DISEASE DIAGNOSIS — TEST RUNNER")
    print(f"{'='*72}")

    if args.validate or args.all:
        rc = run_phase2_validation()
        total_failures += rc

    if args.unit or args.all:
        rc = run_unit_tests()
        total_failures += rc

    if args.integration or args.all:
        rc = run_integration_tests()
        total_failures += rc

    if args.regression or args.all:
        rc = run_regression_tests()
        total_failures += rc

    print(f"\n{SEPARATOR}")
    if total_failures == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {total_failures} test group(s) FAILED")
    print(f"{SEPARATOR}\n")

    return total_failures


if __name__ == "__main__":
    sys.exit(main())
