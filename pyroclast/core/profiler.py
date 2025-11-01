#!/usr/bin/env python3
"""
Layer-by-Layer Profiler for Pyroclast

Profile PyTorch models to identify bottlenecks and optimization opportunities.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from pyroclast.utils import Colors, print_success, print_error, print_step


class LayerProfile:
    """Profile data for a single layer"""

    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.total_time_ms = 0.0
        self.min_time_ms = float("inf")
        self.max_time_ms = 0.0
        self.cpu_time_ms = 0.0
        self.cuda_time_ms = 0.0
        self.memory_mb = 0.0

    def add_sample(
        self, time_ms: float, cpu_time: float = 0, cuda_time: float = 0, memory: float = 0
    ):
        """Add a timing sample"""
        self.count += 1
        self.total_time_ms += time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
        self.cpu_time_ms += cpu_time
        self.cuda_time_ms += cuda_time
        self.memory_mb += memory

    @property
    def avg_time_ms(self) -> float:
        """Average time per invocation"""
        return self.total_time_ms / self.count if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "count": self.count,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms if self.min_time_ms != float("inf") else 0.0,
            "max_time_ms": self.max_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "cuda_time_ms": self.cuda_time_ms,
            "memory_mb": self.memory_mb,
        }


class ModelProfiler:
    """Profile PyTorch model layer-by-layer"""

    def __init__(
        self,
        model_path: str,
        runs: int = 100,
        warmup: int = 10,
        device: str = "cpu",
        output_file: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.runs = runs
        self.warmup = warmup
        self.device = device
        self.output_file = Path(output_file) if output_file else None
        self.verbose = verbose

        self.model = None
        self.layer_profiles: Dict[str, LayerProfile] = {}
        self.total_time_ms = 0.0

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    def load_model(self):
        """Load PyTorch model"""
        print_step(f"Loading model: {self.model_path.name}")

        try:
            # Try loading as HuggingFace model first
            if self.model_path.is_dir():
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(str(self.model_path))
            else:
                # Try loading as PyTorch checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.model = checkpoint["model"]
                elif isinstance(checkpoint, nn.Module):
                    self.model = checkpoint
                else:
                    # Try loading as TorchScript
                    self.model = torch.jit.load(str(self.model_path), map_location=self.device)

            self.model.eval()
            self.model.to(self.device)

            print_success("Model loaded successfully")

        except Exception as e:
            print_error(f"Failed to load model: {e}")
            raise

    def profile_with_pytorch(self) -> None:
        """Profile using PyTorch profiler"""
        print_step(f"Profiling with {self.runs} runs ({self.warmup} warmup)...")

        # Create dummy input (this is simplified - real implementation needs proper inputs)
        # For now, we'll profile the forward pass structure
        try:
            # Get input shape from model if possible
            if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
                batch_size = 1
                seq_len = 128
                hidden_size = self.model.config.hidden_size
                dummy_input = torch.randn(batch_size, seq_len, hidden_size).to(self.device)
            else:
                # Generic input
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

            # Warmup
            if self.verbose:
                print(f"  Running {self.warmup} warmup iterations...")
            with torch.no_grad():
                for _ in range(self.warmup):
                    _ = self.model(dummy_input)

            # Profile
            if self.verbose:
                print(f"  Profiling {self.runs} iterations...")

            with torch.no_grad():
                with profile(
                    activities=[ProfilerActivity.CPU]
                    + ([ProfilerActivity.CUDA] if self.device == "cuda" else []),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    for _ in range(self.runs):
                        _ = self.model(dummy_input)

            # Parse profiler results
            self._parse_profiler_results(prof)

            print_success("Profiling completed")

        except Exception as e:
            print_error(f"Profiling failed: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()

    def _parse_profiler_results(self, prof):
        """Parse PyTorch profiler results"""
        # Get key averages
        key_averages = prof.key_averages(group_by_input_shape=False)

        # Group by operation name
        for event in key_averages:
            if event.key in ["aten::to", "aten::empty", "aten::zero_"]:
                continue  # Skip overhead operations

            name = event.key
            cpu_time = event.cpu_time_total / 1000  # Convert to ms
            cuda_time = event.cuda_time_total / 1000 if hasattr(event, "cuda_time_total") else 0
            memory = (
                event.cpu_memory_usage / (1024 * 1024) if hasattr(event, "cpu_memory_usage") else 0
            )  # MB

            total_time = cpu_time + cuda_time

            if name not in self.layer_profiles:
                self.layer_profiles[name] = LayerProfile(name)

            self.layer_profiles[name].add_sample(
                time_ms=total_time, cpu_time=cpu_time, cuda_time=cuda_time, memory=memory
            )

        # Calculate total time
        self.total_time_ms = sum(p.total_time_ms for p in self.layer_profiles.values())

    def print_results(self):
        """Print profiling results"""
        if not self.layer_profiles:
            print(f"{Colors.YELLOW}No profiling data available{Colors.RESET}")
            return

        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}LAYER PERFORMANCE BREAKDOWN{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

        # Sort by total time
        sorted_profiles = sorted(
            self.layer_profiles.values(), key=lambda p: p.total_time_ms, reverse=True
        )

        # Show top layers
        top_n = min(20, len(sorted_profiles))
        print(f"Top {top_n} operations by time:\n")

        for profile in sorted_profiles[:top_n]:
            percentage = (
                (profile.total_time_ms / self.total_time_ms * 100) if self.total_time_ms > 0 else 0
            )

            # Create bar chart
            bar_width = int(percentage / 2)  # Scale to fit terminal
            bar = "█" * bar_width

            # Color code by percentage
            if percentage > 20:
                color = Colors.RED
            elif percentage > 10:
                color = Colors.YELLOW
            else:
                color = Colors.GREEN

            # Truncate long names
            display_name = profile.name[:40] + "..." if len(profile.name) > 40 else profile.name

            print(
                f"{display_name:<45} {color}{bar}{Colors.RESET} {percentage:5.1f}% ({profile.total_time_ms:.2f}ms)"
            )

        print(f"\n{Colors.DIM}Total profiled time: {self.total_time_ms:.2f}ms{Colors.RESET}\n")

        # Suggestions
        self._print_suggestions(sorted_profiles)

    def _print_suggestions(self, sorted_profiles: List[LayerProfile]):
        """Print optimization suggestions"""
        if not sorted_profiles:
            return

        print(f"{Colors.CYAN}{'─'*70}{Colors.RESET}")
        print(f"{Colors.CYAN}💡 Optimization Suggestions:{Colors.RESET}\n")

        bottleneck = sorted_profiles[0]
        percentage = (
            (bottleneck.total_time_ms / self.total_time_ms * 100) if self.total_time_ms > 0 else 0
        )

        if percentage > 20:
            print(
                f"  • {Colors.YELLOW}{bottleneck.name}{Colors.RESET} is the main bottleneck ({percentage:.1f}% of time)"
            )

            # Operation-specific suggestions
            if "conv" in bottleneck.name.lower():
                print(f"    → Consider: Depthwise separable convolutions or MobileNet architecture")
            elif "linear" in bottleneck.name.lower() or "matmul" in bottleneck.name.lower():
                print(f"    → Consider: Quantization or pruning to reduce computation")
            elif "attention" in bottleneck.name.lower():
                print(f"    → Consider: Flash Attention or approximate attention mechanisms")

        if self.device == "cpu":
            print(f"  • Running on CPU - try CUDA for potential speedup")
            print(f"    → python3 pyroclast.py profile {self.model_path} --device cuda")

        print(f"  • Try auto-optimization to find the best backend:")
        print(f"    → python3 pyroclast.py optimize {self.model_path} -o outputs/")

        print()

    def save_results(self):
        """Save profiling results to JSON"""
        if not self.output_file:
            return

        print_step(f"Saving results to: {self.output_file}")

        results = {
            "model": str(self.model_path),
            "runs": self.runs,
            "device": self.device,
            "total_time_ms": self.total_time_ms,
            "layers": [
                p.to_dict()
                for p in sorted(
                    self.layer_profiles.values(), key=lambda p: p.total_time_ms, reverse=True
                )
            ],
        }

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, "w") as f:
            json.dump(results, f, indent=2)

        print_success("Results saved")

    def run(self) -> int:
        """Run profiling"""
        try:
            self.load_model()
            self.profile_with_pytorch()
            self.print_results()

            if self.output_file:
                self.save_results()

            return 0

        except Exception as e:
            print_error(f"Profiling failed: {e}")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Profile PyTorch model layer-by-layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic profiling
  python3 pyroclast.py profile models/my_model

  # With more runs for accuracy
  python3 pyroclast.py profile models/my_model --runs 1000

  # Save to file
  python3 pyroclast.py profile models/my_model -o profile.json

  # CUDA profiling
  python3 pyroclast.py profile models/my_model --device cuda
        """,
    )

    parser.add_argument("model_path", help="Path to PyTorch model")

    parser.add_argument("-r", "--runs", type=int, default=100, help="Number of profiling runs")

    parser.add_argument("-w", "--warmup", type=int, default=10, help="Number of warmup runs")

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to profile on",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Save results to JSON file",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    profiler = ModelProfiler(
        model_path=args.model_path,
        runs=args.runs,
        warmup=args.warmup,
        device=args.device,
        output_file=args.output,
        verbose=args.verbose,
    )

    exit_code = profiler.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
