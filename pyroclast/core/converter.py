#!/usr/bin/env python3
"""
Executorch Model Converter
Converts HuggingFace models (especially ParlerTTS) to Executorch format
Supports multiple backends: CPU (XNNPACK), Vulkan, and portable
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from parler_tts import ParlerTTSForConditionalGeneration

# Executorch imports
try:
    from executorch.exir import to_edge
    from executorch.exir.backend.backend_api import to_backend
    from torch.export import export
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig

    EXECUTORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Executorch not fully available: {e}")
    EXECUTORCH_AVAILABLE = False


from pyroclast.utils import (
    Colors,
    print_header,
    print_subheader,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_step,
    load_input_file,
    validate_input_spec,
    InputSpec,
)


class ModelInfo:
    """Stores metadata about the model"""

    def __init__(self):
        self.model_type: str = ""
        self.param_count: int = 0
        self.model_size_mb: float = 0.0
        self.architecture: str = ""
        self.components: Dict[str, Any] = {}
        self.input_shapes: Dict[str, Tuple] = {}


class ExecutorchConverter:
    """Converts models to Executorch format"""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        backend: str = "xnnpack",
        quantize: bool = False,
        verbose: bool = False,
        input_file: Optional[str] = None,
        description: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.backend = backend
        self.quantize = quantize
        self.verbose = verbose
        self.input_file = input_file
        self.description = description
        self.prompt = prompt
        self.model_info = ModelInfo()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> Tuple[torch.nn.Module, Any]:
        """Load the model from HuggingFace"""
        print_subheader("Loading Model")
        print_step(f"Loading from: {self.model_path}")

        start_time = time.time()

        # Try to detect model type
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.model_info.model_type = config.get("model_type", "unknown")
                self.model_info.architecture = config.get("architectures", ["unknown"])[0]

        # Load model based on type
        try:
            if self.model_info.model_type == "parler_tts":
                print_info("Model Type", "ParlerTTS (Text-to-Speech)")
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    str(self.model_path), torch_dtype=torch.float32
                )
                tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            else:
                print_info("Model Type", "Generic Transformers Model")
                model = AutoModel.from_pretrained(str(self.model_path))
                tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            raise

        model.eval()

        # Calculate model statistics
        self.model_info.param_count = sum(p.numel() for p in model.parameters())
        self.model_info.model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        load_time = time.time() - start_time

        print_success(f"Model loaded in {load_time:.2f}s")
        print_info("Parameters", f"{self.model_info.param_count:,}")
        print_info("Model Size", f"{self.model_info.model_size_mb:.2f} MB")
        print_info("Architecture", self.model_info.architecture)

        return model, tokenizer

    def analyze_model(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze model structure"""
        print_subheader("Analyzing Model Structure")

        analysis = {"modules": {}, "layers": 0, "parameters_by_layer": {}}

        # Count different module types
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in analysis["modules"]:
                analysis["modules"][module_type] = 0
            analysis["modules"][module_type] += 1

        # Count layers
        analysis["layers"] = sum(1 for _ in model.modules())

        # Show top module types
        print_step("Module Breakdown:")
        sorted_modules = sorted(analysis["modules"].items(), key=lambda x: x[1], reverse=True)[:10]

        for module_type, count in sorted_modules:
            print(f"  • {module_type}: {count}")

        return analysis

    def create_example_inputs(self, model: torch.nn.Module, tokenizer: Any) -> Tuple:
        """Create example inputs for export"""
        print_subheader("Preparing Example Inputs")

        # Load inputs from file or CLI args or use defaults
        input_spec = self._get_input_spec()

        if self.model_info.model_type == "parler_tts":
            # ParlerTTS requires two text inputs
            description = input_spec.description or "A female speaker with a clear voice"
            prompt = input_spec.prompt or "Hello world"

            print_info("Description", f'"{description}"')
            print_info("Prompt", f'"{prompt}"')

            input_ids = tokenizer(description, return_tensors="pt").input_ids
            prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            print_info("Description tokens", input_ids.shape)
            print_info("Prompt tokens", prompt_input_ids.shape)

            return (input_ids, prompt_input_ids)
        else:
            # Generic transformer input
            text = input_spec.prompt or input_spec.description or "This is a test input"
            print_info("Input text", f'"{text}"')

            inputs = tokenizer(text, return_tensors="pt")
            return tuple(inputs.values())

    def _get_input_spec(self) -> InputSpec:
        """Get input specification from file or CLI args"""
        # Priority: input_file > CLI args > defaults
        if self.input_file:
            try:
                spec = load_input_file(self.input_file)
                if isinstance(spec, list):
                    print_warning(f"Batch input file detected, using first input only")
                    spec = spec[0]
                return spec
            except Exception as e:
                print_error(f"Failed to load input file: {e}")
                raise

        # Use CLI args if provided
        return InputSpec(description=self.description, prompt=self.prompt)

    def export_to_executorch(self, model: torch.nn.Module, example_inputs: Tuple) -> bytes:
        """Export model to Executorch format"""
        print_subheader(f"Exporting to Executorch ({self.backend})")

        if not EXECUTORCH_AVAILABLE:
            print_error("Executorch is not properly installed")
            raise RuntimeError("Executorch not available")

        start_time = time.time()

        try:
            # Step 1: Capture the model graph
            print_step("Step 1/4: Capturing model graph...")
            with torch.no_grad():
                exported_program = export(model, example_inputs)
            print_success("Graph captured")

            # Step 2: Convert to Edge dialect
            print_step("Step 2/4: Converting to Edge dialect...")
            edge_config = EdgeCompileConfig(
                _check_ir_validity=False,  # May need to disable for complex models
            )
            edge_program = to_edge(exported_program, compile_config=edge_config)
            print_success("Edge dialect generated")

            # Step 3: Apply backend-specific optimizations
            print_step(f"Step 3/4: Applying {self.backend} optimizations...")

            if self.backend == "xnnpack":
                try:
                    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                        XnnpackPartitioner,
                    )

                    edge_program = edge_program.to_backend(XnnpackPartitioner())
                    print_success("XNNPACK partitioner applied")
                except Exception as e:
                    print_error(f"XNNPACK optimization failed: {e}")
                    print_info("Note", "Continuing with portable backend")
            elif self.backend == "vulkan":
                try:
                    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
                        VulkanPartitioner,
                    )

                    edge_program = edge_program.to_backend(VulkanPartitioner())
                    print_success("Vulkan partitioner applied")
                except Exception as e:
                    print_error(f"Vulkan optimization failed: {e}")
                    print_info("Note", "Continuing with portable backend")

            # Step 4: Generate Executorch program
            print_step("Step 4/4: Generating Executorch program...")
            exec_config = ExecutorchBackendConfig(
                extract_delegate_segments=True,
            )
            executorch_program = edge_program.to_executorch(config=exec_config)
            print_success("Executorch program generated")

            export_time = time.time() - start_time
            print_success(f"Export completed in {export_time:.2f}s")

            return executorch_program.buffer

        except Exception as e:
            print_error(f"Export failed: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            raise

    def save_model(self, buffer: bytes, metadata: Dict[str, Any]):
        """Save the exported model and metadata"""
        print_subheader("Saving Model")

        # Save .pte file
        model_name = self.model_path.name
        output_file = self.output_dir / f"{model_name}_{self.backend}.pte"

        print_step(f"Writing model to: {output_file}")
        with open(output_file, "wb") as f:
            f.write(buffer)

        file_size_mb = len(buffer) / (1024 * 1024)
        print_success(f"Model saved ({file_size_mb:.2f} MB)")

        # Save metadata
        metadata_file = self.output_dir / f"{model_name}_{self.backend}_metadata.json"
        metadata.update(
            {
                "model_path": str(self.model_path),
                "backend": self.backend,
                "output_file": str(output_file),
                "file_size_mb": file_size_mb,
                "param_count": self.model_info.param_count,
                "original_size_mb": self.model_info.model_size_mb,
            }
        )

        print_step(f"Writing metadata to: {metadata_file}")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print_success("Metadata saved")

        return output_file

    def convert(self) -> Path:
        """Main conversion pipeline"""
        print_header("EXECUTORCH MODEL CONVERTER")

        total_start = time.time()

        try:
            # Load model
            model, tokenizer = self.load_model()

            # Analyze model
            analysis = self.analyze_model(model)

            # Create example inputs
            example_inputs = self.create_example_inputs(model, tokenizer)

            # Export to Executorch
            buffer = self.export_to_executorch(model, example_inputs)

            # Save
            output_file = self.save_model(buffer, analysis)

            total_time = time.time() - total_start

            print_subheader("Conversion Complete!")
            print_success(f"Total time: {total_time:.2f}s")
            print_info("Output", output_file)

            return Path(output_file)

        except Exception as e:
            print_error(f"Conversion failed: {e}")
            raise


def main():
    """Parses CLI arguments and calls the converter"""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to Executorch format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert ParlerTTS model with XNNPACK (CPU) backend
  %(prog)s models/eclipse_code -o outputs/

  # Convert with Vulkan (GPU) backend
  %(prog)s models/eclipse_code -o outputs/ --backend vulkan

  # Enable verbose output
  %(prog)s models/eclipse_code -o outputs/ -v
        """,
    )

    parser.add_argument("model_path", help="Path to HuggingFace model directory")

    parser.add_argument(
        "-o",
        "--output-dir",
        default="./outputs",
        help="Output directory for converted model (default: ./outputs)",
    )

    parser.add_argument(
        "-b",
        "--backend",
        choices=["xnnpack", "vulkan", "portable"],
        default="xnnpack",
        help="Target backend (default: xnnpack for CPU)",
    )

    parser.add_argument(
        "-q", "--quantize", action="store_true", help="Apply quantization (reduces model size)"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    # Input specification options
    input_group = parser.add_argument_group("input specification")
    input_group.add_argument(
        "--input-file",
        help="JSON file containing input specification (description, prompt, metadata)",
    )
    input_group.add_argument(
        "--description", help="Description text (for ParlerTTS voice description or model input)"
    )
    input_group.add_argument("--prompt", help="Prompt text (for ParlerTTS text or model input)")

    args = parser.parse_args()

    converter = ExecutorchConverter(
        model_path=args.model_path,
        output_dir=args.output_dir,
        backend=args.backend,
        quantize=args.quantize,
        verbose=args.verbose,
        input_file=args.input_file,
        description=args.description,
        prompt=args.prompt,
    )

    try:
        converter.convert()
        sys.exit(0)
    except Exception as e:
        print_error(f"\nConversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
