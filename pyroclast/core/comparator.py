#!/usr/bin/env python3
"""
Executorch Model Comparison Tool
Compare performance across different backends and models
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import subprocess

import torch
import numpy as np


from pyroclast.utils import (
    Colors,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_step,
    format_latency,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""

    backend: str
    model_name: str
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput: float
    model_size_mb: float
    successful_runs: int
    total_runs: int


def format_latency(ms: float) -> str:
    """Format latency with units"""
    if ms < 1:
        return f"{ms*1000:.2f}μs"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"


class ModelComparator:
    """Compare different models and backends"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def benchmark_pytorch_model(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> BenchmarkResult:
        """Benchmark original PyTorch model"""
        print_step("Benchmarking PyTorch (baseline)...")

        model.eval()
        times = []

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)

        # Benchmark
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(input_tensor)
                times.append(time.time() - start)

        times_ms = [t * 1000 for t in times]

        # Calculate model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        result = BenchmarkResult(
            backend="PyTorch (CPU)",
            model_name="baseline",
            mean_latency_ms=np.mean(times_ms),
            median_latency_ms=np.median(times_ms),
            std_latency_ms=np.std(times_ms),
            min_latency_ms=np.min(times_ms),
            max_latency_ms=np.max(times_ms),
            p95_latency_ms=np.percentile(times_ms, 95),
            p99_latency_ms=np.percentile(times_ms, 99),
            throughput=1.0 / np.mean(times),
            model_size_mb=model_size,
            successful_runs=num_runs,
            total_runs=num_runs,
        )

        print_success(f"Mean latency: {format_latency(result.mean_latency_ms)}")
        self.results.append(result)
        return result

    def benchmark_executorch_model(
        self,
        model_path: Path,
        backend_name: str,
        input_tensor: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> BenchmarkResult:
        """Benchmark Executorch model"""
        print_step(f"Benchmarking Executorch ({backend_name})...")

        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        model = _load_for_executorch(str(model_path))
        times = []

        # Warmup
        for _ in range(warmup_runs):
            try:
                _ = model.run_method("forward", (input_tensor,))
            except:
                pass

        # Benchmark
        successful = 0
        for _ in range(num_runs):
            try:
                start = time.time()
                _ = model.run_method("forward", (input_tensor,))
                times.append(time.time() - start)
                successful += 1
            except:
                pass

        if not times:
            print_error("All runs failed")
            return None

        times_ms = [t * 1000 for t in times]
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        result = BenchmarkResult(
            backend=f"Executorch ({backend_name})",
            model_name=model_path.stem,
            mean_latency_ms=np.mean(times_ms),
            median_latency_ms=np.median(times_ms),
            std_latency_ms=np.std(times_ms),
            min_latency_ms=np.min(times_ms),
            max_latency_ms=np.max(times_ms),
            p95_latency_ms=np.percentile(times_ms, 95),
            p99_latency_ms=np.percentile(times_ms, 99),
            throughput=1.0 / np.mean(times),
            model_size_mb=model_size_mb,
            successful_runs=successful,
            total_runs=num_runs,
        )

        print_success(f"Mean latency: {format_latency(result.mean_latency_ms)}")
        self.results.append(result)
        return result

    def print_comparison_table(self):
        """Print comparison table"""
        if not self.results:
            print_warning("No results to compare")
            return

        print_header("BENCHMARK COMPARISON")

        # Define table structure
        columns = [
            "Backend",
            "Mean Latency",
            "Median",
            "P95",
            "P99",
            "Throughput",
            "Size",
            "Success Rate",
        ]
        widths = [20, 14, 14, 14, 14, 14, 12, 14]

        header = ""
        separator = ""
        for col, width in zip(columns, widths):
            header += f"{Colors.BOLD}{Colors.CYAN}{col:^{width}}{Colors.END} │ "
            separator += "─" * width + "─┼─"

        print(header[:-3])
        print(f"{Colors.CYAN}{separator[:-3]}{Colors.END}")

        # Find baseline for speedup calculation
        baseline = next((r for r in self.results if "PyTorch" in r.backend), self.results[0])

        # Sort by mean latency
        sorted_results = sorted(self.results, key=lambda x: x.mean_latency_ms)

        for result in sorted_results:
            is_best = result == sorted_results[0]

            values = [
                result.backend,
                format_latency(result.mean_latency_ms),
                format_latency(result.median_latency_ms),
                format_latency(result.p95_latency_ms),
                format_latency(result.p99_latency_ms),
                f"{result.throughput:.2f}/s",
                f"{result.model_size_mb:.1f}MB",
                f"{result.successful_runs}/{result.total_runs}",
            ]

            row = ""
            color = Colors.GREEN if is_best else ""
            for val, width in zip(values, widths):
                row += f"{color}{val:^{width}}{Colors.END} │ "
            print(row[:-3])

        # Print speedup summary
        print(f"\n{Colors.BOLD}{Colors.HEADER}Performance vs Baseline:{Colors.END}")
        for result in sorted_results:
            if result != baseline:
                if baseline.mean_latency_ms == 0 or result.mean_latency_ms == 0:
                    speedup = "N/A"
                else:
                    speedup_val = baseline.mean_latency_ms / result.mean_latency_ms
                    if speedup_val > 1:
                        speedup = f"{Colors.GREEN}{speedup_val:.2f}x faster{Colors.END}"
                    else:
                        speedup = f"{Colors.RED}{1/speedup_val:.2f}x slower{Colors.END}"
                print(f"  {result.backend:30s} {speedup}")

    def save_results(self, output_path: Path):
        """Save results to JSON"""
        data = {
            "results": [asdict(r) for r in self.results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print_success(f"Results saved to {output_path}")

    def generate_report(self):
        """Generate a text report"""
        print_header("SUMMARY REPORT")

        if not self.results:
            print("No results available")
            return

        best = min(self.results, key=lambda x: x.mean_latency_ms)
        worst = max(self.results, key=lambda x: x.mean_latency_ms)

        print(f"{Colors.BOLD}Best Performance:{Colors.END}")
        print(f"  Backend: {Colors.GREEN}{best.backend}{Colors.END}")
        print(f"  Mean Latency: {Colors.GREEN}{format_latency(best.mean_latency_ms)}{Colors.END}")
        print(f"  Throughput: {Colors.GREEN}{best.throughput:.2f} inferences/sec{Colors.END}")

        print(f"\n{Colors.BOLD}Worst Performance:{Colors.END}")
        print(f"  Backend: {Colors.RED}{worst.backend}{Colors.END}")
        print(f"  Mean Latency: {Colors.RED}{format_latency(worst.mean_latency_ms)}{Colors.END}")
        print(f"  Throughput: {Colors.RED}{worst.throughput:.2f} inferences/sec{Colors.END}")

        if best != worst:
            improvement = worst.mean_latency_ms / best.mean_latency_ms
            print(
                f"\n{Colors.BOLD}Maximum Speedup:{Colors.END} {Colors.GREEN}{improvement:.2f}x{Colors.END}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and Executorch model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare PyTorch baseline with Executorch models
  %(prog)s --baseline mobilenet_v2 --executorch outputs/*.pte

  # Save results to JSON
  %(prog)s --baseline mobilenet_v2 --executorch outputs/*.pte --save results.json

  # Custom number of runs
  %(prog)s --baseline mobilenet_v2 --executorch outputs/*.pte --runs 500
        """,
    )

    parser.add_argument("--baseline", help="PyTorch model for baseline (e.g., 'mobilenet_v2')")

    parser.add_argument("--executorch", nargs="+", help="Executorch .pte model files to benchmark")

    parser.add_argument(
        "--runs", type=int, default=100, help="Number of benchmark runs (default: 100)"
    )

    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup runs (default: 10)"
    )

    parser.add_argument("--save", type=Path, help="Save results to JSON file")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.baseline and not args.executorch:
        parser.print_help()
        sys.exit(1)

    comparator = ModelComparator(verbose=args.verbose)

    # Benchmark baseline
    if args.baseline:
        print_header(f"Loading Baseline Model: {args.baseline}")
        try:
            model = torch.hub.load("pytorch/vision:v0.10.0", args.baseline, pretrained=True)
            model.eval()
            input_tensor = torch.randn(1, 3, 224, 224)

            comparator.benchmark_pytorch_model(model, input_tensor, args.runs, args.warmup)
        except Exception as e:
            print_error(f"Failed to load baseline: {e}")

    # Benchmark Executorch models
    if args.executorch:
        input_tensor = torch.randn(1, 3, 224, 224)

        for model_path_str in args.executorch:
            model_path = Path(model_path_str)
            if not model_path.exists():
                print_warning(f"Skipping {model_path} (not found)")
                continue

            print_header(f"Testing {model_path.name}")

            # Detect backend from filename
            backend = "portable"
            if "xnnpack" in model_path.stem:
                backend = "XNNPACK"
            elif "vulkan" in model_path.stem:
                backend = "Vulkan"

            try:
                comparator.benchmark_executorch_model(
                    model_path, backend, input_tensor, args.runs, args.warmup
                )
            except Exception as e:
                print_error(f"Benchmark failed: {e}")

    # Print comparison
    comparator.print_comparison_table()
    comparator.generate_report()

    # Save results
    if args.save:
        comparator.save_results(args.save)

    print(f"\n{Colors.BOLD}{Colors.GREEN}All benchmarks complete!{Colors.END}\n")


if __name__ == "__main__":
    main()
