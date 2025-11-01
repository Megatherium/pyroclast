#!/usr/bin/env python3
"""
Batch Testing for Pyroclast

Run multiple test cases from a JSON file and generate a report.
Perfect for regression testing and validation.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from pyroclast.utils import Colors, print_success, print_error, print_step, print_warning


class TestCase:
    """Represents a single test case"""

    def __init__(
        self,
        name: str,
        inputs: List[List[float]],
        input_shape: Optional[List[int]] = None,
        expected_output: Optional[List] = None,
        expected_shape: Optional[List[int]] = None,
        tolerance: float = 1e-5,
    ):
        self.name = name
        self.inputs = inputs
        self.input_shape = input_shape
        self.expected_output = expected_output
        self.expected_shape = expected_shape
        self.tolerance = tolerance
        self.passed = False
        self.actual_output = None
        self.actual_shape = None
        self.error = None
        self.inference_time_ms = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "name": self.name,
            "passed": self.passed,
            "inference_time_ms": self.inference_time_ms,
            "input_shape": self.input_shape,
            "actual_shape": self.actual_shape,
            "expected_shape": self.expected_shape,
            "error": self.error,
        }


class BatchTester:
    """Run batch tests on an Executorch model"""

    def __init__(
        self,
        model_path: str,
        test_file: str,
        output_file: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.test_file = Path(test_file)
        self.output_file = Path(output_file) if output_file else None
        self.verbose = verbose

        self.model = None
        self.test_cases: List[TestCase] = []
        self.results: Dict[str, Any] = {}

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not self.test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

    def load_model(self):
        """Load the Executorch model"""
        print_step(f"Loading model: {self.model_path.name}")

        try:
            from executorch.extension.pybindings.portable_lib import (
                _load_for_executorch,
            )

            self.model = _load_for_executorch(str(self.model_path))
            print_success("Model loaded successfully")

        except Exception as e:
            print_error(f"Failed to load model: {e}")
            raise

    def load_tests(self):
        """Load test cases from JSON file"""
        print_step(f"Loading tests: {self.test_file.name}")

        try:
            with open(self.test_file) as f:
                data = json.load(f)

            # Validate format
            if "tests" not in data:
                raise ValueError("Test file must contain 'tests' array")

            if not isinstance(data["tests"], list):
                raise ValueError("'tests' must be an array")

            # Load each test case
            for i, test_data in enumerate(data["tests"]):
                if not isinstance(test_data, dict):
                    raise ValueError(f"Test {i} must be a dictionary")

                test_case = TestCase(
                    name=test_data.get("name", f"Test {i+1}"),
                    inputs=test_data.get("inputs"),
                    input_shape=test_data.get("input_shape"),
                    expected_output=test_data.get("expected_output"),
                    expected_shape=test_data.get("expected_shape"),
                    tolerance=test_data.get("tolerance", 1e-5),
                )

                if not test_case.inputs:
                    raise ValueError(f"Test '{test_case.name}' missing 'inputs' field")

                self.test_cases.append(test_case)

            print_success(f"Loaded {len(self.test_cases)} test cases")

        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in test file: {e}")
            raise
        except Exception as e:
            print_error(f"Failed to load tests: {e}")
            raise

    def run_tests(self):
        """Run all test cases"""
        print_step("Running tests...")

        passed = 0
        failed = 0

        for i, test in enumerate(self.test_cases, 1):
            if self.verbose:
                print(f"\n{Colors.CYAN}[{i}/{len(self.test_cases)}] {test.name}{Colors.RESET}")

            try:
                # Prepare input
                input_tensor = torch.tensor(test.inputs, dtype=torch.float32)

                if test.input_shape:
                    input_tensor = input_tensor.reshape(test.input_shape)

                # Run inference
                start = time.time()
                outputs = self.model.forward((input_tensor,))
                test.inference_time_ms = (time.time() - start) * 1000

                # Extract output
                if isinstance(outputs, (list, tuple)):
                    output_tensor = outputs[0]
                else:
                    output_tensor = outputs

                test.actual_output = output_tensor.detach().cpu().numpy().tolist()
                test.actual_shape = list(output_tensor.shape)

                # Validate shape if expected
                if test.expected_shape:
                    if test.actual_shape != test.expected_shape:
                        test.passed = False
                        test.error = f"Shape mismatch: expected {test.expected_shape}, got {test.actual_shape}"
                        failed += 1
                        if self.verbose:
                            print_warning(test.error)
                        continue

                # Validate output if expected
                if test.expected_output is not None:
                    expected = np.array(test.expected_output)
                    actual = np.array(test.actual_output)

                    if not np.allclose(actual, expected, atol=test.tolerance):
                        test.passed = False
                        max_diff = np.max(np.abs(actual - expected))
                        test.error = f"Output mismatch: max difference {max_diff:.6f} (tolerance {test.tolerance})"
                        failed += 1
                        if self.verbose:
                            print_warning(test.error)
                        continue

                # Test passed
                test.passed = True
                passed += 1

                if self.verbose:
                    print_success(f"Passed ({test.inference_time_ms:.2f}ms)")

            except Exception as e:
                test.passed = False
                test.error = str(e)
                failed += 1

                if self.verbose:
                    print_error(f"Failed: {e}")

        # Store results
        self.results = {
            "model": str(self.model_path),
            "total_tests": len(self.test_cases),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(self.test_cases) * 100) if self.test_cases else 0,
            "tests": [test.to_dict() for test in self.test_cases],
        }

        return passed, failed

    def print_summary(self):
        """Print test summary"""
        passed = self.results["passed"]
        failed = self.results["failed"]
        total = self.results["total_tests"]

        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}TEST SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

        print(f"  Total:   {total}")
        print(f"  {Colors.GREEN}Passed:  {passed}{Colors.RESET}")

        if failed > 0:
            print(f"  {Colors.RED}Failed:  {failed}{Colors.RESET}")
        else:
            print(f"  Failed:  {failed}")

        print(f"  Success: {self.results['success_rate']:.1f}%\n")

        # Show failed tests
        if failed > 0:
            print(f"{Colors.RED}Failed Tests:{Colors.RESET}")
            for test in self.test_cases:
                if not test.passed:
                    print(f"  ✗ {test.name}")
                    if test.error:
                        print(f"    {Colors.DIM}{test.error}{Colors.RESET}")
            print()

    def save_results(self):
        """Save results to JSON file"""
        if not self.output_file:
            return

        print_step(f"Saving results to: {self.output_file}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print_success("Results saved")

    def run(self) -> int:
        """Run full batch test suite"""
        try:
            self.load_model()
            self.load_tests()
            passed, failed = self.run_tests()
            self.print_summary()

            if self.output_file:
                self.save_results()

            # Return exit code: 0 if all passed, 1 if any failed
            return 0 if failed == 0 else 1

        except Exception as e:
            print_error(f"Batch test failed: {e}")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run batch tests on Executorch model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test File Format:
{
  "tests": [
    {
      "name": "Test 1",
      "inputs": [[1.0, 2.0, 3.0]],
      "input_shape": [1, 3],
      "expected_output": [[0.5, 0.3, 0.2]],
      "expected_shape": [1, 3],
      "tolerance": 1e-5
    },
    {
      "name": "Test 2",
      "inputs": [[4.0, 5.0, 6.0]],
      "input_shape": [1, 3]
    }
  ]
}

Examples:
  # Run tests
  python3 pyroclast.py batch-test model.pte tests.json

  # With output file
  python3 pyroclast.py batch-test model.pte tests.json -o results.json

  # Verbose mode
  python3 pyroclast.py batch-test model.pte tests.json -v
        """,
    )

    parser.add_argument("model_path", help="Path to Executorch model (.pte file)")

    parser.add_argument("test_file", help="Path to test cases JSON file")

    parser.add_argument(
        "-o",
        "--output",
        help="Save results to JSON file",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    tester = BatchTester(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output,
        verbose=args.verbose,
    )

    exit_code = tester.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
