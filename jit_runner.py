#!/usr/bin/env python3
"""
PyTorch JIT Runner
Run inference on TorchScript models with CPU/GPU/Vulkan backends
Includes comprehensive benchmarking
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch


from etorch_utils import (
    Colors,
    print_header,
    print_subheader,
    print_success,
    print_warning,
    print_error,
    print_info,
)


def check_vulkan_support() -> bool:
    """Check if Vulkan backend is available"""
    try:
        return hasattr(torch.backends, 'vulkan') and torch.backends.vulkan.is_available()
    except:
        return False


class BenchmarkStats:
    """Statistics for benchmarking"""
    def __init__(self):
        self.times: List[float] = []

    def add_run(self, duration: float):
        self.times.append(duration)

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


class JITRunner:
    """Runner for TorchScript models"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        verbose: bool = False
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.verbose = verbose
        self.model = None
        self.metadata = {}

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load metadata if available
        metadata_path = self.model_path.with_name(
            self.model_path.stem + "_metadata.json"
        )
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

    def check_backends(self):
        """Check available backends"""
        print_subheader("Backend Availability")

        backends = {
            "CPU": True,
            "CUDA": torch.cuda.is_available(),
            "MPS": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "Vulkan": check_vulkan_support()
        }

        for backend, available in backends.items():
            status = f"{Colors.GREEN}✓{Colors.END}" if available else f"{Colors.RED}✗{Colors.END}"
            print(f"  {backend:10s} {status}")

        # Validate requested device
        if self.device == "cuda" and not backends["CUDA"]:
            print_warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        if self.device == "vulkan" and not backends["Vulkan"]:
            print_warning("Vulkan requested but not available, falling back to CPU")
            self.device = "cpu"

        print_info("Selected Device", self.device.upper())

        return backends

    def load_model(self):
        """Load TorchScript model"""
        print_subheader("Loading TorchScript Model")
        print_info("Path", self.model_path)

        start_time = time.time()

        try:
            # Load model with map_location
            self.model = torch.jit.load(
                str(self.model_path),
                map_location=torch.device(self.device)
            )
            self.model.eval()

            load_time = time.time() - start_time

            print_success(f"Model loaded in {load_time:.3f}s")

            # Print metadata if available
            if self.metadata:
                print_info("Method", self.metadata.get("method", "unknown"))
                print_info("Optimized", self.metadata.get("optimized", False))
                file_size_mb = self.metadata.get("file_size_mb", 0)
                print_info("Model Size", f"{file_size_mb:.2f} MB")

        except Exception as e:
            print_error(f"Failed to load model: {e}")
            raise

    def create_dummy_input(self):
        """Create dummy input tensor"""
        # For now, create standard vision input
        # TODO: Detect input shape from metadata
        input_tensor = torch.randn(1, 3, 224, 224)

        if self.device == "cuda":
            input_tensor = input_tensor.cuda()

        return input_tensor

    def run_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run single inference"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            with torch.no_grad():
                output = self.model(input_tensor)
            return output
        except Exception as e:
            print_error(f"Inference failed: {e}")
            raise

    def benchmark(
        self,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> BenchmarkStats:
        """Benchmark model performance"""
        print_subheader(f"Benchmarking ({num_runs} runs, {warmup_runs} warmup)")

        stats = BenchmarkStats()

        # Create input
        input_tensor = self.create_dummy_input()

        # Warmup
        print(f"  {Colors.YELLOW}Warming up...{Colors.END}", end=" ", flush=True)
        for _ in range(warmup_runs):
            try:
                self.run_inference(input_tensor)
            except:
                pass

        # Sync if CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()

        print(f"{Colors.GREEN}✓{Colors.END}")

        # Benchmark
        print(f"  {Colors.YELLOW}Running benchmark...{Colors.END}", end=" ", flush=True)
        successful_runs = 0

        for i in range(num_runs):
            try:
                start_time = time.time()
                self.run_inference(input_tensor)

                # Sync if CUDA
                if self.device == "cuda":
                    torch.cuda.synchronize()

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
        warmup_runs: int = 10
    ):
        """Main execution flow"""
        print_header("PYTORCH JIT RUNNER")

        # Check backends
        self.check_backends()

        # Load model
        self.load_model()

        if benchmark:
            self.benchmark(num_runs, warmup_runs)
        else:
            print_subheader("Running Single Inference")
            input_tensor = self.create_dummy_input()
            print_info("Input Shape", str(tuple(input_tensor.shape)))

            start = time.time()
            output = self.run_inference(input_tensor)
            duration = time.time() - start

            print_info("Output Shape", str(tuple(output.shape)))
            print_info("Inference Time", f"{duration*1000:.2f}", "ms")

        print_header("Complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on TorchScript models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  %(prog)s model_jit.pt

  # Benchmark on CPU
  %(prog)s model_jit.pt --benchmark --runs 1000

  # Run on CUDA
  %(prog)s model_jit.pt --device cuda --benchmark

  # Run on Vulkan (if available)
  %(prog)s model_jit.pt --device vulkan --benchmark
        """
    )

    parser.add_argument(
        "model_path",
        help="Path to TorchScript .pt model file"
    )

    parser.add_argument(
        "-d", "--device",
        choices=["cpu", "cuda", "vulkan"],
        default="cpu",
        help="Device to run on (default: cpu)"
    )

    parser.add_argument(
        "-b", "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )

    parser.add_argument(
        "-r", "--runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)"
    )

    parser.add_argument(
        "-w", "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs (default: 10)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    try:
        runner = JITRunner(
            model_path=args.model_path,
            device=args.device,
            verbose=args.verbose
        )

        runner.run(
            benchmark=args.benchmark,
            num_runs=args.runs,
            warmup_runs=args.warmup
        )

        sys.exit(0)

    except Exception as e:
        print_error(f"Runner failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
