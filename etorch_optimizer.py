#!/usr/bin/env python3
"""
Executorch Auto-Optimizer
Automatically tests all available backends and selects the best one
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil

import torch
from transformers import AutoModel, AutoTokenizer

from etorch_utils import (
    Colors,
    print_header,
    print_subheader,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_step,
)

# Import converters
from etorch_converter import ExecutorchConverter, EXECUTORCH_AVAILABLE


class BackendResult:
    """Results from testing a single backend"""

    def __init__(self, backend: str):
        self.backend = backend
        self.conversion_success = False
        self.conversion_time = 0.0
        self.conversion_error = None
        self.model_path = None
        self.model_size_mb = 0.0

        # Benchmark results
        self.benchmark_success = False
        self.mean_latency_ms = 0.0
        self.median_latency_ms = 0.0
        self.min_latency_ms = 0.0
        self.max_latency_ms = 0.0
        self.p95_latency_ms = 0.0
        self.p99_latency_ms = 0.0
        self.throughput = 0.0
        self.benchmark_error = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            "backend": self.backend,
            "conversion": {
                "success": self.conversion_success,
                "time_seconds": self.conversion_time,
                "error": self.conversion_error,
                "model_size_mb": self.model_size_mb,
            },
            "benchmark": {
                "success": self.benchmark_success,
                "mean_latency_ms": self.mean_latency_ms,
                "median_latency_ms": self.median_latency_ms,
                "min_latency_ms": self.min_latency_ms,
                "max_latency_ms": self.max_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "p99_latency_ms": self.p99_latency_ms,
                "throughput_per_sec": self.throughput,
                "error": self.benchmark_error,
            },
        }


class AutoOptimizer:
    """Automatically optimize model for best backend"""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        backends: Optional[List[str]] = None,
        runs: int = 50,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.runs = runs
        self.verbose = verbose
        self.results: List[BackendResult] = []

        # Determine which backends to test
        if backends:
            self.backends = backends
        else:
            # Default: test all available
            self.backends = ["portable", "xnnpack"]
            # TODO: Add vulkan detection

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_backend(self, backend: str) -> BackendResult:
        """Test a single backend"""
        print_subheader(f"Testing Backend: {backend.upper()}")
        result = BackendResult(backend)

        # Create temporary directory for this backend's output
        temp_dir = tempfile.mkdtemp(prefix=f"etorch_{backend}_")

        try:
            # Step 1: Convert model
            print_step(f"Converting model with {backend} backend...")
            start_time = time.time()

            try:
                converter = ExecutorchConverter(
                    model_path=str(self.model_path),
                    output_dir=temp_dir,
                    backend=backend,
                    verbose=self.verbose,
                )
                output_file = converter.convert()
                result.conversion_time = time.time() - start_time
                result.conversion_success = True
                result.model_path = output_file
                result.model_size_mb = output_file.stat().st_size / (1024 * 1024)

                print_success(
                    f"Conversion successful ({result.conversion_time:.2f}s, {result.model_size_mb:.2f}MB)"
                )

            except Exception as e:
                result.conversion_time = time.time() - start_time
                result.conversion_error = str(e)
                print_error(f"Conversion failed: {e}")
                return result

            # Step 2: Benchmark model
            print_step(f"Benchmarking {backend} model ({self.runs} runs)...")

            try:
                # Import here to avoid circular dependency
                from etorch_runner import ExecutorchRunner

                runner = ExecutorchRunner(str(result.model_path), verbose=self.verbose)
                runner.load_model()

                # Create example inputs (simplified for now)
                # Note: ExecutorchRunner.benchmark expects List[torch.Tensor]
                # TODO: Support model-specific inputs
                example_inputs = [torch.randn(1, 3, 224, 224)]

                # Run benchmark (num_runs parameter, not runs)
                bench_stats = runner.benchmark(example_inputs, num_runs=self.runs, warmup_runs=5)

                # Get summary statistics
                stats = bench_stats.summary()

                if stats:
                    result.benchmark_success = True
                    result.mean_latency_ms = stats["mean_ms"]
                    result.median_latency_ms = stats["median_ms"]
                    result.min_latency_ms = stats["min_ms"]
                    result.max_latency_ms = stats["max_ms"]
                    result.p95_latency_ms = stats["p95_ms"]
                    result.p99_latency_ms = stats["p99_ms"]
                    result.throughput = stats["throughput"]

                    print_success(
                        f"Benchmark complete: {result.mean_latency_ms:.2f}ms mean latency"
                    )
                else:
                    raise Exception("Benchmark returned no stats")

            except Exception as e:
                result.benchmark_error = str(e)
                print_error(f"Benchmark failed: {e}")
                if self.verbose:
                    import traceback

                    traceback.print_exc()
                return result

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        return result

    def run_optimization(self) -> Optional[BackendResult]:
        """Run optimization across all backends"""
        print_header("EXECUTORCH AUTO-OPTIMIZER")

        print_info("Model", str(self.model_path))
        print_info("Backends to test", ", ".join(self.backends))
        print_info("Benchmark runs", str(self.runs))

        # Test each backend
        for backend in self.backends:
            result = self.test_backend(backend)
            self.results.append(result)

        # Analyze results
        return self.analyze_results()

    def analyze_results(self) -> Optional[BackendResult]:
        """Analyze results and determine winner"""
        print_subheader("Analysis")

        # Filter to successful results
        successful = [r for r in self.results if r.conversion_success and r.benchmark_success]

        if not successful:
            print_error("No backends succeeded in both conversion and benchmarking")
            return None

        # Sort by mean latency (lower is better)
        successful.sort(key=lambda r: r.mean_latency_ms)
        winner = successful[0]

        # Print comparison table
        print("\n" + "=" * 80)
        print(f"{Colors.BOLD}Results Comparison:{Colors.END}\n")

        # Table header
        header = f"{'Backend':<12} {'Latency':<12} {'Throughput':<15} {'Size':<10} {'Status':<10}"
        print(header)
        print("-" * 80)

        for result in self.results:
            if result.conversion_success and result.benchmark_success:
                is_winner = result.backend == winner.backend
                prefix = f"{Colors.GREEN}✓" if is_winner else " "
                suffix = f" {'BEST' if is_winner else ''}{Colors.END}"

                latency = f"{result.mean_latency_ms:.2f}ms"
                throughput = f"{result.throughput:.2f}/sec"
                size = f"{result.model_size_mb:.2f}MB"
                status = "✓"

                print(
                    f"{prefix} {result.backend:<11} {latency:<12} {throughput:<15} {size:<10} {status:<10}{suffix}"
                )
            elif result.conversion_success:
                print(
                    f"  {result.backend:<11} {'N/A':<12} {'N/A':<15} {f'{result.model_size_mb:.2f}MB':<10} ✗ Bench"
                )
            else:
                print(f"  {result.backend:<11} {'N/A':<12} {'N/A':<15} {'N/A':<10} ✗ Convert")

        print("=" * 80 + "\n")

        # Show winner details
        if len(successful) > 1:
            speedup = successful[-1].mean_latency_ms / winner.mean_latency_ms
            print_success(f"Winner: {winner.backend.upper()}")
            print_info("Mean Latency", f"{winner.mean_latency_ms:.2f}ms")
            print_info("Throughput", f"{winner.throughput:.2f} inferences/sec")
            print_info(
                "Speedup",
                f"{speedup:.2f}x faster than {successful[-1].backend}",
            )
        else:
            print_success(f"Only backend that succeeded: {winner.backend.upper()}")
            print_info("Mean Latency", f"{winner.mean_latency_ms:.2f}ms")
            print_info("Throughput", f"{winner.throughput:.2f} inferences/sec")

        return winner

    def save_results(self, winner: Optional[BackendResult]) -> Path:
        """Save results to JSON file"""
        print_subheader("Saving Results")

        results_data = {
            "model_path": str(self.model_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backends_tested": len(self.backends),
            "benchmark_runs": self.runs,
            "winner": winner.backend if winner else None,
            "results": [r.to_dict() for r in self.results],
        }

        output_file = self.output_dir / "optimization_results.json"
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print_success(f"Results saved to: {output_file}")
        return output_file

    def save_winner_model(self, winner: BackendResult) -> Path:
        """Save the winning model to output directory"""
        if not winner:
            print_warning("No winner to save")
            return None

        print_subheader("Saving Optimized Model")

        model_name = self.model_path.name
        print_step(f"Converting with {winner.backend} backend...")

        # Re-convert with winner backend to final location
        try:
            converter = ExecutorchConverter(
                model_path=str(self.model_path),
                output_dir=str(self.output_dir),
                backend=winner.backend,
                verbose=False,  # Less verbose for final conversion
            )
            final_model = converter.convert()
            print_success(f"Optimized model saved: {final_model}")
            return final_model
        except Exception as e:
            print_error(f"Failed to save optimized model: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Auto-optimize Executorch model by testing all backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-optimize a model (tests all backends)
  %(prog)s models/my_model -o outputs/

  # Test specific backends
  %(prog)s models/my_model -o outputs/ --backends portable xnnpack

  # More benchmark runs for accuracy
  %(prog)s models/my_model -o outputs/ --runs 200
        """,
    )

    parser.add_argument("model_path", help="Path to model directory")

    parser.add_argument(
        "-o",
        "--output-dir",
        default="./outputs",
        help="Output directory (default: ./outputs)",
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["portable", "xnnpack", "vulkan"],
        help="Backends to test (default: all available)",
    )

    parser.add_argument(
        "--runs", type=int, default=50, help="Benchmark runs per backend (default: 50)"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not EXECUTORCH_AVAILABLE:
        print_error("Executorch is not available")
        sys.exit(1)

    optimizer = AutoOptimizer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        backends=args.backends,
        runs=args.runs,
        verbose=args.verbose,
    )

    try:
        winner = optimizer.run_optimization()

        if winner:
            optimizer.save_results(winner)
            optimizer.save_winner_model(winner)

            print_subheader("Optimization Complete!")
            print_info("Best Backend", winner.backend.upper())
            print_info("Mean Latency", f"{winner.mean_latency_ms:.2f}ms")

            sys.exit(0)
        else:
            print_error("Optimization failed - no successful backends")
            optimizer.save_results(None)
            sys.exit(1)

    except Exception as e:
        print_error(f"Optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
