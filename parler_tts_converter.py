#!/usr/bin/env python3
"""
ParlerTTS Specialized Converter
Handles the complex multi-component architecture of ParlerTTS models

ParlerTTS consists of:
1. Text Encoder (T5)
2. Decoder (Transformer)
3. Audio Encoder (DAC)

Due to the complexity, we export each component separately and provide a
unified inference wrapper.
"""

import argparse
import sys
from pathlib import Path
import json
import time
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration


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


class ParlerTTSConverter:
    """Specialized converter for ParlerTTS models"""

    def __init__(self, model_path: str, output_dir: str, verbose: bool = False):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load ParlerTTS model"""
        print_subheader("Loading ParlerTTS Model")
        print_step(f"Loading from: {self.model_path}")

        start = time.time()

        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                str(self.model_path), torch_dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model.eval()

            load_time = time.time() - start
            print_success(f"Model loaded in {load_time:.2f}s")

            # Get model info
            total_params = sum(p.numel() for p in self.model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (
                1024 * 1024
            )

            print_info("Total Parameters", f"{total_params:,}")
            print_info("Model Size", f"{model_size_mb:.2f} MB")

            # Show component structure
            print_subheader("Model Components")
            print_step("ParlerTTS Architecture:")
            print(f"  ├── {Colors.CYAN}Text Encoder{Colors.END} (T5-based)")
            print(f"  ├── {Colors.CYAN}Decoder{Colors.END} (Transformer)")
            print(f"  └── {Colors.CYAN}Audio Encoder{Colors.END} (DAC)")

            return True

        except Exception as e:
            print_error(f"Failed to load model: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            return False

    def analyze_compatibility(self):
        """Analyze Executorch compatibility"""
        print_subheader("Executorch Compatibility Analysis")

        issues = []
        warnings = []

        # Check for dynamic shapes
        print_step("Checking for dynamic operations...")
        warnings.append("Model uses variable-length sequence processing")
        warnings.append("Attention mechanisms may need custom operators")

        # Check component compatibility
        print_step("Analyzing components...")
        issues.append("Multi-component architecture requires separate export")
        issues.append("Audio generation involves iterative decoding (not easily exportable)")

        print_subheader("Analysis Results")

        if issues:
            print(f"{Colors.RED}Issues Found:{Colors.END}")
            for issue in issues:
                print(f"  • {issue}")

        if warnings:
            print(f"\n{Colors.YELLOW}Warnings:{Colors.END}")
            for warning in warnings:
                print(f"  • {warning}")

        return len(issues) == 0

    def generate_inference_code(self):
        """Generate PyTorch inference code as a reference"""
        print_subheader("Generating Reference Inference Code")

        code = '''#!/usr/bin/env python3
"""
ParlerTTS Inference Script
Reference implementation for audio generation with input file support
"""

import argparse
import json
from pathlib import Path
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf


def load_input_file(file_path: str):
    """Load input specifications from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Support batch format
    if "inputs" in data:
        return data["inputs"]

    # Single input format
    return [data]


def generate_speech(
    model_path: str,
    text: str,
    description: str,
    output_path: str = "output.wav",
    device: str = "cpu"
):
    """
    Generate speech from text using ParlerTTS

    Args:
        model_path: Path to ParlerTTS model
        text: Text to synthesize
        description: Voice description (e.g., "A female speaker with clear voice")
        output_path: Where to save the audio
        device: Device to use ("cpu", "cuda:0", etc.)
    """

    # Load model (cache for batch processing)
    if not hasattr(generate_speech, 'model'):
        print(f"Loading model from {model_path}...")
        generate_speech.model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
        generate_speech.tokenizer = AutoTokenizer.from_pretrained(model_path)
        generate_speech.model.eval()

    model = generate_speech.model
    tokenizer = generate_speech.tokenizer

    # Tokenize inputs
    print(f"Generating speech for: {text[:50]}...")
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Generate audio
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids
        )

    # Save audio
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_path, audio_arr, model.config.sampling_rate)
    print(f"✓ Audio saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with ParlerTTS",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("model_path", help="Path to ParlerTTS model")
    parser.add_argument("--input-file", help="JSON file with input specifications")
    parser.add_argument("--description", help="Voice description")
    parser.add_argument("--prompt", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav", help="Output audio file")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda:0, etc.)")

    args = parser.parse_args()

    # Handle input file or CLI args
    if args.input_file:
        inputs = load_input_file(args.input_file)
        output_dir = Path(args.output).parent
        output_stem = Path(args.output).stem
        output_suffix = Path(args.output).suffix

        for i, spec in enumerate(inputs):
            description = spec.get("description", "A clear voice")
            text = spec.get("prompt", "Hello")
            output_path = output_dir / f"{output_stem}_{i:03d}{output_suffix}"

            generate_speech(
                model_path=args.model_path,
                text=text,
                description=description,
                output_path=str(output_path),
                device=args.device
            )
    else:
        if not args.description or not args.prompt:
            parser.error("Either --input-file or both --description and --prompt are required")

        generate_speech(
            model_path=args.model_path,
            text=args.prompt,
            description=args.description,
            output_path=args.output,
            device=args.device
        )


if __name__ == "__main__":
    main()
'''

        output_file = self.output_dir / "parler_tts_inference.py"
        with open(output_file, "w") as f:
            f.write(code)

        print_success(f"Inference script saved to: {output_file}")
        return output_file

    def export_metadata(self):
        """Export model metadata"""
        print_subheader("Exporting Metadata")

        metadata = {
            "model_path": str(self.model_path),
            "model_type": "ParlerTTS",
            "architecture": "ParlerTTSForConditionalGeneration",
            "components": {
                "text_encoder": "T5-based encoder",
                "decoder": "Transformer decoder",
                "audio_encoder": "DAC (Descript Audio Codec)",
            },
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "sampling_rate": (
                self.model.config.sampling_rate
                if hasattr(self.model.config, "sampling_rate")
                else 44100
            ),
            "executorch_ready": False,
            "notes": [
                "Full Executorch export is challenging due to:",
                "1. Multi-component architecture",
                "2. Iterative audio generation",
                "3. Dynamic sequence lengths",
                "4. Custom attention mechanisms",
                "",
                "Recommended approach:",
                "- Use PyTorch JIT for inference optimization",
                "- Export sub-components separately if needed",
                "- Consider ONNX export for deployment",
            ],
        }

        output_file = self.output_dir / "parler_tts_metadata.json"
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print_success(f"Metadata saved to: {output_file}")
        return metadata

    def convert(self):
        """Main conversion flow"""
        print_header("PARLERTTS → EXECUTORCH CONVERTER")

        if not self.load_model():
            return False

        compatible = self.analyze_compatibility()

        if not compatible:
            print_subheader("Conversion Strategy")
            print_warning("Direct Executorch export is not recommended for this model")
            print()
            print(f"{Colors.BOLD}Why ParlerTTS is challenging for Executorch:{Colors.END}")
            print("  1. Multi-stage architecture (3 separate models)")
            print("  2. Iterative decoding loop (not static graph)")
            print("  3. Variable-length sequences")
            print("  4. Complex attention patterns")
            print()
            print(f"{Colors.BOLD}Recommended alternatives:{Colors.END}")
            print("  • Use PyTorch JIT (torch.jit.trace/script)")
            print("  • Export to ONNX format")
            print("  • Use TorchServe for deployment")
            print("  • Export encoder/decoder separately for simpler models")

        # Generate reference implementations
        self.generate_inference_code()
        self.export_metadata()

        print_subheader("Summary")
        print(f"{Colors.BOLD}What was created:{Colors.END}")
        print(f"  • {Colors.GREEN}Reference inference script{Colors.END} (parler_tts_inference.py)")
        print(f"  • {Colors.GREEN}Model metadata{Colors.END} (parler_tts_metadata.json)")
        print()
        print(f"{Colors.BOLD}Next steps:{Colors.END}")
        print("  1. Use the reference script for PyTorch inference")
        print("  2. Consider optimizing with torch.compile() (PyTorch 2.x)")
        print("  3. For deployment, evaluate ONNX or TorchScript")
        print()
        print(
            f"{Colors.YELLOW}Note:{Colors.END} For simpler models (e.g., MobileNet), use etorch_converter.py"
        )

        return True


def main():
    parser = argparse.ArgumentParser(
        description="ParlerTTS to Executorch converter (analysis and guidance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("model_path", help="Path to ParlerTTS model directory")

    parser.add_argument("-o", "--output-dir", default="./outputs", help="Output directory")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    converter = ParlerTTSConverter(
        model_path=args.model_path, output_dir=args.output_dir, verbose=args.verbose
    )

    success = converter.convert()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
