#!/usr/bin/env python3
"""
Executorch Toolkit - Master CLI
Unified interface for model conversion, running, analysis, and comparison
"""

import argparse
import sys
from pathlib import Path
import subprocess


from etorch_utils import (
    Colors,
    print_header,
    print_step,
    print_dim,
)


def print_banner():
    """Print cool ASCII banner"""
    banner = f"""{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ███████╗██╗  ██╗███████╗ ██████╗██╗   ██╗████████╗ ██████╗ ██████╗     ║
║   ██╔════╝╚██╗██╔╝██╔════╝██╔════╝██║   ██║╚══██╔══╝██╔═══██╗██╔══██╗    ║
║   █████╗   ╚███╔╝ █████╗  ██║     ██║   ██║   ██║   ██║   ██║██████╔╝    ║
║   ██╔══╝   ██╔██╗ ██╔══╝  ██║     ██║   ██║   ██║   ██║   ██║██╔══██╗    ║
║   ███████╗██╔╝ ██╗███████╗╚██████╗╚██████╔╝   ██║   ╚██████╔╝██║  ██║    ║
║   ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝    ║
║                                                                          ║
║                          TOOLKIT v1.0                                    ║
║                  Convert • Run • Analyze • Compare                       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
{Colors.END}"""
    print(banner)


def run_command(cmd: list, description: str) -> int:
    """Run a subprocess command"""
    print_step(description)
    print_dim(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)
    return result.returncode


def convert_model(args) -> int:
    """Convert model to Executorch"""
    cmd = [
        sys.executable,
        "etorch_converter.py",
        args.model_path,
        "-o",
        args.output_dir,
        "-b",
        args.backend,
    ]

    if args.quantize:
        cmd.append("-q")

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "Converting model to Executorch")


def run_model(args) -> int:
    """Run inference on model"""
    cmd = [
        sys.executable,
        "etorch_runner.py",
        args.model_path,
    ]

    if args.benchmark:
        cmd.append("--benchmark")
        cmd.extend(["--runs", str(args.runs)])
        cmd.extend(["--warmup", str(args.warmup)])

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "Running model inference")


def analyze_model(args) -> int:
    """Analyze model architecture"""
    cmd = [
        sys.executable,
        "model_analyzer.py",
        args.model_path,
        "--max-depth",
        str(args.max_depth),
    ]

    if args.save_report:
        cmd.append("--save-report")

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "Analyzing model structure")


def compare_models(args) -> int:
    """Compare model performance"""
    cmd = [
        sys.executable,
        "etorch_compare.py",
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
    ]

    if args.baseline:
        cmd.extend(["--baseline", args.baseline])

    if args.executorch:
        cmd.append("--executorch")
        cmd.extend(args.executorch)

    if args.save:
        cmd.extend(["--save", args.save])

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "Comparing model performance")


def optimize_model(args) -> int:
    """Auto-optimize model by testing all backends"""
    cmd = [
        sys.executable,
        "etorch_optimizer.py",
        args.model_path,
        "-o",
        args.output_dir,
        "--runs",
        str(args.runs),
    ]

    if args.backends:
        cmd.extend(["--backends"] + args.backends)

    if args.verbose:
        cmd.append("-v")

    return run_command(cmd, "Auto-optimizing model")


def main():
    parser = argparse.ArgumentParser(
        description="Executorch Toolkit - Unified CLI for model operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Colors.BOLD}Examples:{Colors.END}
  # Convert a model
  %(prog)s convert models/eclipse_code -o outputs/ --backend xnnpack

  # Run inference with benchmarking
  %(prog)s run outputs/model.pte --benchmark --runs 1000

  # Analyze model architecture
  %(prog)s analyze models/eclipse_code --max-depth 4

  # Compare PyTorch vs Executorch
  %(prog)s compare --baseline mobilenet_v2 --executorch outputs/*.pte

  # Auto-optimize (test all backends)
  %(prog)s optimize models/my_model -o outputs/ --runs 100

{Colors.BOLD}Commands:{Colors.END}
  convert    Convert HuggingFace/PyTorch model to Executorch format
  run        Run inference on Executorch model
  analyze    Analyze model architecture and parameters
  compare    Compare performance across backends
  optimize   Auto-optimize model by testing all backends

{Colors.BOLD}For detailed help on a command:{Colors.END}
  %(prog)s <command> --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model to Executorch format")
    convert_parser.add_argument("model_path", help="Path to model")
    convert_parser.add_argument("-o", "--output-dir", default="./outputs", help="Output directory")
    convert_parser.add_argument(
        "-b",
        "--backend",
        default="xnnpack",
        choices=["xnnpack", "vulkan", "portable"],
        help="Backend",
    )
    convert_parser.add_argument("-q", "--quantize", action="store_true", help="Apply quantization")
    convert_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    convert_parser.set_defaults(func=convert_model)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run inference on Executorch model")
    run_parser.add_argument("model_path", help="Path to .pte model file")
    run_parser.add_argument("-b", "--benchmark", action="store_true", help="Run benchmark")
    run_parser.add_argument("-r", "--runs", type=int, default=100, help="Number of runs")
    run_parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup runs")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    run_parser.set_defaults(func=run_model)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze model architecture")
    analyze_parser.add_argument("model_path", help="Path to model")
    analyze_parser.add_argument("--max-depth", type=int, default=3, help="Max tree depth")
    analyze_parser.add_argument("--save-report", action="store_true", help="Save JSON report")
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    analyze_parser.set_defaults(func=analyze_model)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare model performance")
    compare_parser.add_argument("--baseline", help="PyTorch baseline model")
    compare_parser.add_argument("--executorch", nargs="+", help="Executorch models to compare")
    compare_parser.add_argument("-r", "--runs", type=int, default=100, help="Number of runs")
    compare_parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup runs")
    compare_parser.add_argument("--save", help="Save results to JSON")
    compare_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    compare_parser.set_defaults(func=compare_models)

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize", help="Auto-optimize by testing all backends"
    )
    optimize_parser.add_argument("model_path", help="Path to model")
    optimize_parser.add_argument("-o", "--output-dir", default="./outputs", help="Output directory")
    optimize_parser.add_argument(
        "--backends",
        nargs="+",
        choices=["portable", "xnnpack", "vulkan"],
        help="Backends to test (default: all available)",
    )
    optimize_parser.add_argument(
        "-r", "--runs", type=int, default=50, help="Benchmark runs per backend"
    )
    optimize_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    optimize_parser.set_defaults(func=optimize_model)

    # JIT Convert command
    jit_convert_parser = subparsers.add_parser("jit-convert", help="Convert model to TorchScript")
    jit_convert_parser.add_argument("model_path", help="Path to model")
    jit_convert_parser.add_argument(
        "-o", "--output-dir", default="./outputs", help="Output directory"
    )
    jit_convert_parser.add_argument(
        "-m", "--method", choices=["script", "trace"], default="trace", help="Conversion method"
    )
    jit_convert_parser.add_argument(
        "--no-optimize", action="store_true", help="Disable optimization"
    )
    jit_convert_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    jit_convert_parser.set_defaults(
        func=lambda args: run_command(
            [
                sys.executable,
                "jit_converter.py",
                args.model_path,
                "-o",
                args.output_dir,
                "-m",
                args.method,
            ]
            + (["--no-optimize"] if args.no_optimize else [])
            + (["-v"] if args.verbose else []),
            "Converting to TorchScript",
        )
    )

    # JIT Run command
    jit_run_parser = subparsers.add_parser("jit-run", help="Run TorchScript model")
    jit_run_parser.add_argument("model_path", help="Path to .pt model")
    jit_run_parser.add_argument(
        "-d", "--device", choices=["cpu", "cuda", "vulkan"], default="cpu", help="Device"
    )
    jit_run_parser.add_argument("-b", "--benchmark", action="store_true", help="Run benchmark")
    jit_run_parser.add_argument("-r", "--runs", type=int, default=100, help="Number of runs")
    jit_run_parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup runs")
    jit_run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    jit_run_parser.set_defaults(
        func=lambda args: run_command(
            [sys.executable, "jit_runner.py", args.model_path, "-d", args.device]
            + (
                ["--benchmark", "--runs", str(args.runs), "--warmup", str(args.warmup)]
                if args.benchmark
                else []
            )
            + (["-v"] if args.verbose else []),
            "Running TorchScript model",
        )
    )

    # TTS command
    tts_parser = subparsers.add_parser("tts", help="Text-to-Speech with ParlerTTS")
    tts_parser.add_argument("model_path", help="Path to ParlerTTS model")
    tts_parser.add_argument("text", help="Text to synthesize")
    tts_parser.add_argument("-o", "--output", default="output", help="Output path")
    tts_parser.add_argument(
        "--description",
        default="A clear female voice speaks with moderate speed and pitch.",
        help="Voice description",
    )
    tts_parser.add_argument(
        "--formats",
        nargs="+",
        choices=["wav", "opus", "ogg", "mp3"],
        default=["wav"],
        help="Output formats",
    )
    tts_parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cpu", help="Device")
    tts_parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    tts_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    tts_parser.set_defaults(
        func=lambda args: run_command(
            [
                sys.executable,
                "parler_tts_jit.py",
                args.model_path,
                args.text,
                "-o",
                args.output,
                "--description",
                args.description,
                "--formats",
            ]
            + args.formats
            + ["-d", args.device]
            + (["--compile"] if args.compile else [])
            + (["-v"] if args.verbose else []),
            "Generating speech",
        )
    )

    args = parser.parse_args()

    if not args.command:
        print_banner()
        parser.print_help()
        sys.exit(0)

    print_banner()

    # Execute command
    return_code = args.func(args)
    sys.exit(return_code if return_code is not None else 0)


if __name__ == "__main__":
    main()
