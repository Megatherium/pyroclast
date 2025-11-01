#!/usr/bin/env python3
"""
Executorch Model Runner
Runs inference on Executorch .pte models with benchmarking and analysis
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import torch
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    EXECUTORCH_AVAILABLE = True
except ImportError as e:
    print(f"Error: Executorch not available: {e}")
    EXECUTORCH_AVAILABLE = False


from pyroclast.utils import (
    Colors,
    print_header,
    print_subheader,
    print_success,
    print_warning,
    print_error,
    print_info,
)


class BenchmarkStats:
    """Statistics for benchmarking"""

    def __init__(self):
        self.times: List[float] = []
        self.memory_usage: List[float] = []

    def add_run(self, duration: float, memory_mb: float = 0):
        self.times.append(duration)
        if memory_mb > 0:
            self.memory_usage.append(memory_mb)

    def summary(self) -> Dict[str, float]:
        if not self.times:
            return {}

        times_ms = [t * 1000 for t in self.times]
        return {
            "mean_ms": np.mean(times_ms),
            "median_ms": np.median(times_ms),
            "std_ms": np.std(times_ms),
            "min_ms": np.min(times_ms),
            "max_ms": np.max(times_ms),
            "p95_ms": np.percentile(times_ms, 95),
            "p99_ms": np.percentile(times_ms, 99),
            "throughput": 1.0 / np.mean(self.times) if np.mean(self.times) > 0 else 0,
        }


class ExecutorchRunner:
    """Runner for Executorch models"""

    def __init__(self, model_path: str, verbose: bool = False):
        self.model_path = Path(model_path)
        self.verbose = verbose
        self.model = None
        self.metadata = {}

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load metadata if available
        metadata_path = self.model_path.with_name(self.model_path.stem + "_metadata.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

    def load_model(self):
        """Load the Executorch model"""
        print_subheader("Loading Model")
        print_info("Path", self.model_path)

        start_time = time.time()

        try:
            self.model = _load_for_executorch(str(self.model_path))
            load_time = time.time() - start_time

            print_success(f"Model loaded in {load_time:.3f}s")

            # Print metadata if available
            if self.metadata:
                print_info("Backend", self.metadata.get("backend", "unknown"))
                print_info("Parameters", f"{self.metadata.get('param_count', 0):,}")
                print_info("Model Size", f"{self.metadata.get('file_size_mb', 0):.2f} MB")

        except Exception as e:
            print_error(f"Failed to load model: {e}")
            raise

    def inspect_model(self):
        """Inspect model structure"""
        print_subheader("Model Information")

        try:
            # Get methods
            methods = []
            for i in range(10):  # Try first 10 methods
                try:
                    method_name = self.model.method_name(i)
                    methods.append(method_name)
                except:
                    break

            if methods:
                print_info("Methods", ", ".join(methods))
            else:
                print_warning("No methods found")

        except Exception as e:
            print_warning(f"Could not inspect model: {e}")

    def run_inference(
        self, inputs: List[torch.Tensor], method_name: str = "forward"
    ) -> List[torch.Tensor]:
        """Run inference on the model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            method = self.model.run_method(method_name, tuple(inputs))
            return list(method)
        except Exception as e:
            print_error(f"Inference failed: {e}")
            raise

    def benchmark(
        self,
        inputs: List[torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10,
        method_name: str = "forward",
    ) -> BenchmarkStats:
        """Benchmark model performance"""
        print_subheader(f"Benchmarking ({num_runs} runs, {warmup_runs} warmup)")

        stats = BenchmarkStats()

        # Warmup
        print(f"  {Colors.YELLOW}Warming up...{Colors.END}", end=" ", flush=True)
        for _ in range(warmup_runs):
            try:
                self.run_inference(inputs, method_name)
            except:
                pass
        print(f"{Colors.GREEN}✓{Colors.END}")

        # Benchmark
        print(f"  {Colors.YELLOW}Running benchmark...{Colors.END}", end=" ", flush=True)
        successful_runs = 0

        for i in range(num_runs):
            try:
                start_time = time.time()
                self.run_inference(inputs, method_name)
                duration = time.time() - start_time
                stats.add_run(duration)
                successful_runs += 1
            except Exception as e:
                if self.verbose:
                    print_error(f"Run {i} failed: {e}")

        print(f"{Colors.GREEN}✓{Colors.END}")

        # Print summary
        summary = stats.summary()
        if summary:
            print_subheader("Benchmark Results")
            print_info("Successful runs", f"{successful_runs}/{num_runs}")
            print_info("Mean latency", f"{summary['mean_ms']:.2f}", "ms")
            print_info("Median latency", f"{summary['median_ms']:.2f}", "ms")
            print_info("Std deviation", f"{summary['std_ms']:.2f}", "ms")
            print_info("Min latency", f"{summary['min_ms']:.2f}", "ms")
            print_info("Max latency", f"{summary['max_ms']:.2f}", "ms")
            print_info("P95 latency", f"{summary['p95_ms']:.2f}", "ms")
            print_info("P99 latency", f"{summary['p99_ms']:.2f}", "ms")
            print_info("Throughput", f"{summary['throughput']:.2f}", "inferences/sec")

        return stats

    def run(
        self,
        benchmark: bool = False,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ):
        """Main execution flow"""
        print_header("EXECUTORCH MODEL RUNNER")

        self.load_model()
        self.inspect_model()

        # For now, we'll create dummy inputs
        # In a real scenario, you'd create proper inputs based on model metadata
        print_warning("Using dummy inputs (model-specific inputs not yet implemented)")

        if benchmark:
            dummy_input = [torch.randn(1, 3, 224, 224)]
            self.benchmark(dummy_input, num_runs, warmup_runs)

        print_header("Complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on Executorch models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and inspect a model
  %(prog)s model.pte

  # Run benchmark
  %(prog)s model.pte --benchmark --runs 1000

  # Custom warmup and runs
  %(prog)s model.pte -b -r 500 -w 20
        """,
    )

    parser.add_argument("model_path", help="Path to Executorch .pte model file")

    parser.add_argument("-b", "--benchmark", action="store_true", help="Run performance benchmark")

    parser.add_argument(
        "-r", "--runs", type=int, default=100, help="Number of benchmark runs (default: 100)"
    )

    parser.add_argument(
        "-w", "--warmup", type=int, default=10, help="Number of warmup runs (default: 10)"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if not EXECUTORCH_AVAILABLE:
        print_error("Executorch is not available")
        sys.exit(1)

    try:
        runner = ExecutorchRunner(model_path=args.model_path, verbose=args.verbose)

        runner.run(benchmark=args.benchmark, num_runs=args.runs, warmup_runs=args.warmup)

        sys.exit(0)

    except Exception as e:
        print_error(f"Runner failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
