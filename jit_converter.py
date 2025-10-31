#!/usr/bin/env python3
"""
PyTorch JIT Converter
Converts PyTorch models to TorchScript format for optimized inference
Supports both torch.jit.script and torch.jit.trace
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

from etorch_utils import (
    Colors,
    print_error,
    print_header,
    print_info,
    print_step,
    print_success,
    print_warning,
)


def print_banner():
    """Prints the main banner."""
    print_header("PYTORCH JIT CONVERTER\nTorchScript Optimization Tool", width=70)


def check_vulkan_support() -> bool:
    """Check if Vulkan backend is available"""
    try:
        return hasattr(torch.backends, "vulkan") and torch.backends.vulkan.is_available()
    except Exception:
        return False


class JITConverter:
    """Convert PyTorch models to TorchScript"""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        method: str = "script",
        optimize: bool = True,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.method = method
        self.optimize = optimize
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load PyTorch model"""
        print_header("Loading PyTorch Model", width=70)
        print_step(f"Loading from: {self.model_path}")

        start = time.time()

        try:
            # Try loading as HuggingFace model first
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(str(self.model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            model_type = "HuggingFace"

        except:
            try:
                # Try loading as torch hub model
                model_name = self.model_path.name
                self.model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)
                self.tokenizer = None
                model_type = "TorchHub"
            except Exception:
                try:
                    # Try loading as saved PyTorch model
                    self.model = torch.load(self.model_path)
                    self.tokenizer = None
                    model_type = "PyTorch"
                except Exception as e:
                    print_error(f"Failed to load model: {e}")
                    raise

        self.model.eval()
        load_time = time.time() - start

        # Get model stats
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (
            1024 * 1024
        )

        print_success(f"Model loaded in {load_time:.2f}s")
        print_info("Model Type", model_type)
        print_info("Parameters", f"{total_params:,}")
        print_info("Size", f"{model_size_mb:.2f} MB")

        return self.model

    def check_backends(self):
        """Check available backends"""
        print_header("Backend Availability", width=70)

        backends = {
            "CPU": True,
            "CUDA": torch.cuda.is_available(),
            "MPS": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
            "Vulkan": check_vulkan_support(),
        }

        for backend, available in backends.items():
            status = (
                f"{Colors.GREEN}✓ Available{Colors.END}"
                if available
                else f"{Colors.RED}✗ Not Available{Colors.END}"
            )
            print(f"  {backend:10s} {status}")

        if backends["Vulkan"]:
            print_info("Note", "Vulkan backend ready for GPU acceleration")
        else:
            print_warning("Vulkan not available (requires custom PyTorch build)")

        return backends

    def create_example_inputs(self) -> Tuple:
        """Create example inputs for tracing"""
        print_header("Creating Example Inputs", width=70)

        if self.tokenizer:
            # Text model
            text = "This is a test input for model tracing"
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            example_inputs = tuple(inputs.values())
            print_info("Input Type", "Text (tokenized)")
            print_info("Input Shape", str([t.shape for t in example_inputs]))
        else:
            # Assume vision model
            example_inputs = (torch.randn(1, 3, 224, 224),)
            print_info("Input Type", "Image (224x224)")
            print_info("Input Shape", str(example_inputs[0].shape))

        return example_inputs

    def convert_to_jit(self, example_inputs: Tuple) -> torch.jit.ScriptModule:
        """Convert model to TorchScript"""
        print_header(f"Converting to TorchScript ({self.method})", width=70)

        start = time.time()

        try:
            if self.method == "script":
                print_step("Using torch.jit.script (captures control flow)")
                with torch.no_grad():
                    scripted_model = torch.jit.script(self.model)

            elif self.method == "trace":
                print_step("Using torch.jit.trace (traces execution)")
                with torch.no_grad():
                    scripted_model = torch.jit.trace(self.model, example_inputs)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            conversion_time = time.time() - start
            print_success(f"Conversion completed in {conversion_time:.2f}s")

            # Optimize if requested
            if self.optimize:
                print_step("Optimizing TorchScript model...")
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
                print_success("Optimization applied")

            return scripted_model

        except Exception as e:
            print_error(f"Conversion failed: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            raise

    def save_model(self, scripted_model: torch.jit.ScriptModule):
        """Save TorchScript model"""
        print_header("Saving TorchScript Model", width=70)

        model_name = self.model_path.name if self.model_path.is_dir() else self.model_path.stem
        output_file = self.output_dir / f"{model_name}_jit.pt"

        print_step(f"Saving to: {output_file}")

        start = time.time()
        scripted_model.save(str(output_file))
        save_time = time.time() - start

        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        print_success(f"Model saved in {save_time:.2f}s")
        print_info("File Size", f"{file_size_mb:.2f} MB")
        print_info("Path", str(output_file))

        # Save metadata
        metadata = {
            "model_path": str(self.model_path),
            "output_file": str(output_file),
            "method": self.method,
            "optimized": self.optimize,
            "file_size_mb": file_size_mb,
            "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch_version": torch.__version__,
            "backends_available": {
                "cpu": True,
                "cuda": torch.cuda.is_available(),
                "vulkan": check_vulkan_support(),
            },
        }

        metadata_file = self.output_dir / f"{model_name}_jit_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print_success(f"Metadata saved to: {metadata_file}")

        return output_file

    def verify_model(self, scripted_model: torch.jit.ScriptModule, example_inputs: Tuple):
        """Verify converted model works"""
        print_header("Verifying Converted Model", width=70)

        try:
            print_step("Running inference test...")

            with torch.no_grad():
                # Original model
                original_output = self.model(*example_inputs)

                # Scripted model
                scripted_output = scripted_model(*example_inputs)

            print_success("Inference successful")

            # Compare outputs
            if isinstance(original_output, torch.Tensor) and isinstance(
                scripted_output, torch.Tensor
            ):
                diff = torch.abs(original_output - scripted_output).max().item()
                print_info("Max difference", f"{diff:.6e}")

                if diff < 1e-5:
                    print_success("Outputs match (numerical precision)")
                elif diff < 1e-3:
                    print_warning(f"Small numerical differences detected ({diff:.6e})")
                else:
                    print_warning(f"Larger differences detected ({diff:.6e})")

        except Exception as e:
            print_error(f"Verification failed: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()

    def convert(self):
        """Main conversion pipeline"""
        print_banner()

        try:
            # Load model
            self.load_model()

            # Check backends
            self.check_backends()

            # Create example inputs
            example_inputs = self.create_example_inputs()

            # Convert to JIT
            scripted_model = self.convert_to_jit(example_inputs)

            # Verify
            self.verify_model(scripted_model, example_inputs)

            # Save
            output_file = self.save_model(scripted_model)

            print_header("Conversion Complete!", width=70)
            print_success("TorchScript model ready for inference")
            print_info("Output", output_file)

            print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
            print("  1. Run inference: python3 jit_runner.py <output_file>")
            print("  2. Benchmark: python3 jit_runner.py <output_file> --benchmark")
            print("  3. Compare: python3 etorch.py compare --jit <output_file>")

            return output_file

        except Exception as e:
            print_error(f"Conversion failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to TorchScript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert using torch.jit.script (preserves control flow)
  %(prog)s models/my_model -o outputs/ --method script

  # Convert using torch.jit.trace (traces execution)
  %(prog)s mobilenet_v2 -o outputs/ --method trace

  # Without optimization
  %(prog)s models/my_model -o outputs/ --no-optimize

Methods:
  script  - Compiles model code (handles control flow, slower)
  trace   - Traces execution (faster, no dynamic control flow)
        """,
    )

    parser.add_argument("model_path", help="Path to model directory or name (e.g., 'mobilenet_v2')")

    parser.add_argument(
        "-o", "--output-dir", default="./outputs", help="Output directory (default: ./outputs)"
    )

    parser.add_argument(
        "-m",
        "--method",
        choices=["script", "trace"],
        default="trace",
        help="Conversion method (default: trace)",
    )

    parser.add_argument("--no-optimize", action="store_true", help="Disable optimization")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    converter = JITConverter(
        model_path=args.model_path,
        output_dir=args.output_dir,
        method=args.method,
        optimize=not args.no_optimize,
        verbose=args.verbose,
    )

    try:
        converter.convert()
        sys.exit(0)
    except Exception as e:
        print_error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
