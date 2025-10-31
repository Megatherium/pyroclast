#!/usr/bin/env python3
"""
ParlerTTS JIT Tool
Text-to-Speech with PyTorch JIT optimization
Supports multiple output formats: WAV, Opus, OGG, MP3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
from pydub import AudioSegment
import numpy as np


class Colors:
    """ANSI colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_banner():
    banner = f"""{Colors.BOLD}{Colors.HEADER}
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                      PARLERTTS JIT TOOL                              ║
║                   Text-to-Speech with JIT                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{Colors.END}"""
    print(banner)


def print_section(title: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")


def print_step(msg: str):
    print(f"{Colors.BLUE}▶{Colors.END} {msg}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")


def print_info(key: str, value: str):
    print(f"  {Colors.CYAN}{key}:{Colors.END} {value}")


def convert_audio_format(
    input_wav: Path,
    output_path: Path,
    target_format: str,
    sample_rate: int
) -> Path:
    """Convert audio to target format using ffmpeg via pydub"""

    print_step(f"Converting to {target_format.upper()}...")

    try:
        # Load audio with pydub
        audio = AudioSegment.from_wav(str(input_wav))

        # Set sample rate
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        # Export to target format
        if target_format == "opus":
            # Opus in OGG container
            audio.export(
                str(output_path),
                format="opus",
                codec="libopus",
                bitrate="128k"
            )
        elif target_format == "ogg":
            # Vorbis in OGG container
            audio.export(
                str(output_path),
                format="ogg",
                codec="libvorbis",
                bitrate="192k"
            )
        elif target_format == "mp3":
            # MP3 format
            audio.export(
                str(output_path),
                format="mp3",
                bitrate="192k"
            )
        elif target_format == "wav":
            # WAV format (already have it, just copy)
            audio.export(str(output_path), format="wav")
        else:
            raise ValueError(f"Unsupported format: {target_format}")

        file_size_kb = output_path.stat().st_size / 1024
        print_success(f"{target_format.upper()} file created ({file_size_kb:.1f} KB)")

        return output_path

    except Exception as e:
        print_error(f"Format conversion failed: {e}")
        raise


class ParlerTTSJIT:
    """ParlerTTS with JIT optimization"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        compile: bool = False,
        verbose: bool = False
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.compile = compile
        self.verbose = verbose
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load ParlerTTS model"""
        print_section("Loading ParlerTTS Model")
        print_step(f"Loading from: {self.model_path}")

        start = time.time()

        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start
            print_success(f"Model loaded in {load_time:.2f}s")

            # Get model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print_info("Parameters", f"{total_params:,}")
            print_info("Device", self.device.upper())

            # Optional: Compile with torch.compile (PyTorch 2.0+)
            if self.compile:
                print_step("Compiling model with torch.compile...")
                try:
                    self.model = torch.compile(self.model)
                    print_success("Model compiled")
                except Exception as e:
                    print_warning(f"Compilation failed: {e}")

            return True

        except Exception as e:
            print_error(f"Failed to load model: {e}")
            raise

    def generate_speech(
        self,
        text: str,
        description: str = "A clear female voice speaks with moderate speed and pitch.",
        output_path: Optional[Path] = None,
        output_formats: list = ["wav"]
    ):
        """Generate speech from text"""

        if self.model is None:
            raise RuntimeError("Model not loaded")

        print_section("Generating Speech")
        print_info("Text", text[:100] + ("..." if len(text) > 100 else ""))
        print_info("Description", description[:100])

        # Prepare output path
        if output_path is None:
            output_path = Path("output")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Tokenize inputs
        print_step("Tokenizing inputs...")
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        # Generate audio
        print_step("Generating audio...")
        start = time.time()

        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )

        generation_time = time.time() - start

        # Convert to numpy
        audio_arr = generation.cpu().numpy().squeeze()
        sample_rate = self.model.config.sampling_rate

        duration_sec = len(audio_arr) / sample_rate

        print_success(f"Audio generated in {generation_time:.2f}s")
        print_info("Duration", f"{duration_sec:.2f}s")
        print_info("Sample Rate", f"{sample_rate} Hz")
        print_info("RTF (Real-Time Factor)", f"{generation_time / duration_sec:.2f}x")

        # Save WAV first (intermediate format)
        wav_path = output_path.with_suffix('.wav')
        print_step(f"Saving WAV to: {wav_path}")
        sf.write(str(wav_path), audio_arr, sample_rate)

        wav_size_kb = wav_path.stat().st_size / 1024
        print_success(f"WAV saved ({wav_size_kb:.1f} KB)")

        # Convert to other formats
        output_files = {"wav": wav_path}

        for fmt in output_formats:
            if fmt == "wav":
                continue  # Already saved

            fmt_path = output_path.with_suffix(f'.{fmt}')

            try:
                converted_path = convert_audio_format(
                    wav_path,
                    fmt_path,
                    fmt,
                    sample_rate
                )
                output_files[fmt] = converted_path
            except Exception as e:
                print_warning(f"Failed to convert to {fmt}: {e}")

        # Print summary
        print_section("Output Files")
        for fmt, path in output_files.items():
            size_kb = path.stat().st_size / 1024
            print(f"  {Colors.CYAN}{fmt.upper():6s}{Colors.END} {path} ({size_kb:.1f} KB)")

        return output_files


def main():
    parser = argparse.ArgumentParser(
        description="ParlerTTS Text-to-Speech with JIT optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic TTS
  %(prog)s models/eclipse_code "Hello world" -o output

  # Multiple formats
  %(prog)s models/eclipse_code "Hello" -o output --formats opus ogg mp3 wav

  # Custom voice description
  %(prog)s models/eclipse_code "Test" \\
    --description "A male voice speaks slowly and clearly" \\
    -o output

  # Use CUDA
  %(prog)s models/eclipse_code "Hello" --device cuda -o output

  # With torch.compile (PyTorch 2.0+)
  %(prog)s models/eclipse_code "Hello" --compile -o output
        """
    )

    parser.add_argument(
        "model_path",
        help="Path to ParlerTTS model directory"
    )

    parser.add_argument(
        "text",
        help="Text to synthesize"
    )

    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output file path (without extension, default: output)"
    )

    parser.add_argument(
        "--description",
        default="A clear female voice speaks with moderate speed and pitch.",
        help="Voice description"
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["wav", "opus", "ogg", "mp3"],
        default=["wav"],
        help="Output formats (default: wav)"
    )

    parser.add_argument(
        "-d", "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use (default: cpu)"
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile (PyTorch 2.0+)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print_banner()

    try:
        tts = ParlerTTSJIT(
            model_path=args.model_path,
            device=args.device,
            compile=args.compile,
            verbose=args.verbose
        )

        tts.load_model()

        output_files = tts.generate_speech(
            text=args.text,
            description=args.description,
            output_path=Path(args.output),
            output_formats=args.formats
        )

        print_section("Complete!")
        print_success("Speech generation successful")

        sys.exit(0)

    except Exception as e:
        print_error(f"Failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
