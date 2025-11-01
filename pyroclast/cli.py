"""
Pyroclast CLI - Click-based command-line interface.
"""

import sys
import click
from pathlib import Path

from pyroclast.utils import print_banner


@click.group()
@click.version_option(version="2.0.0", prog_name="Pyroclast")
def cli():
    """
    Pyroclast - Executorch Toolkit

    Convert, run, analyze, and optimize PyTorch models for deployment.
    """
    print_banner()


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="./outputs", help="Output directory")
@click.option(
    "-b",
    "--backend",
    type=click.Choice(["xnnpack", "vulkan", "portable"]),
    default="xnnpack",
    help="Execution backend",
)
@click.option("-q", "--quantize", is_flag=True, help="Apply quantization")
@click.option("--input-file", type=click.Path(exists=True), help="JSON input file")
@click.option("--description", help="Model description (for TTS models)")
@click.option("--prompt", help="Text prompt (for TTS models)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def convert(model_path, output_dir, backend, quantize, input_file, description, prompt, verbose):
    """Convert a model to Executorch format."""
    from pyroclast.core import ExecutorchConverter

    converter = ExecutorchConverter(
        model_path=model_path,
        output_dir=output_dir,
        backend=backend,
        quantize=quantize,
        verbose=verbose,
        input_file=input_file,
        description=description,
        prompt=prompt,
    )

    try:
        model, tokenizer = converter.load_model()
        exported_program = converter.export_model(model, tokenizer)

        if exported_program:
            edge_program = converter.to_edge(exported_program)
            output_path = converter.to_executorch(edge_program)
            click.echo(f"\n✓ Model converted successfully: {output_path}")
            sys.exit(0)
        else:
            click.echo("✗ Conversion failed", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-b", "--benchmark", is_flag=True, help="Run benchmark")
@click.option("-r", "--runs", default=100, help="Number of benchmark runs")
@click.option("-w", "--warmup", default=10, help="Number of warmup runs")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def run(model_path, benchmark, runs, warmup, verbose):
    """Run inference on an Executorch model."""
    from pyroclast.core import ExecutorchRunner

    runner = ExecutorchRunner(model_path, verbose=verbose)

    try:
        if benchmark:
            stats = runner.benchmark(runs=runs, warmup=warmup)
            summary = stats.summary()

            click.echo("\nBenchmark Results:")
            click.echo(f"  Mean:       {summary['mean_ms']:.2f}ms")
            click.echo(f"  Median:     {summary['median_ms']:.2f}ms")
            click.echo(f"  Min:        {summary['min_ms']:.2f}ms")
            click.echo(f"  Max:        {summary['max_ms']:.2f}ms")
            click.echo(f"  P95:        {summary['p95_ms']:.2f}ms")
            click.echo(f"  Throughput: {summary['throughput']:.2f}/sec")
        else:
            output = runner.run()
            click.echo(f"✓ Inference completed")
            if verbose:
                click.echo(f"Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")

        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--max-depth", default=3, help="Maximum tree depth for visualization")
@click.option("--save-report", is_flag=True, help="Save analysis report to JSON")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def analyze(model_path, max_depth, save_report, verbose):
    """Analyze model architecture and parameters."""
    from pyroclast.core import ModelAnalyzer

    analyzer = ModelAnalyzer(model_path, verbose=verbose)

    try:
        analyzer.analyze()

        if save_report:
            report_path = Path(model_path).stem + "_analysis.json"
            analyzer.save_report(report_path)
            click.echo(f"\n✓ Report saved to: {report_path}")

        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--save-report", type=click.Path(), help="Save diagnostic report to JSON file")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def doctor(model_path, save_report, verbose):
    """Check model health and compatibility."""
    from pyroclast.core import ModelDoctor

    doctor = ModelDoctor(model_path, verbose=verbose)

    try:
        report = doctor.diagnose()

        if save_report:
            doctor.save_report(save_report)

        # Exit with error code if critical issues found
        if report["summary"]["errors"] > 0:
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--baseline", type=click.Path(exists=True), help="PyTorch baseline model")
@click.option(
    "--executorch", multiple=True, type=click.Path(exists=True), help="Executorch models to compare"
)
@click.option("-r", "--runs", default=100, help="Number of benchmark runs")
@click.option("-w", "--warmup", default=10, help="Number of warmup runs")
@click.option("--save", type=click.Path(), help="Save results to JSON file")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def compare(baseline, executorch, runs, warmup, save, verbose):
    """Compare performance across models and backends."""
    from pyroclast.core import ModelComparator

    if not baseline and not executorch:
        click.echo("✗ Error: Provide at least one model to compare", err=True)
        sys.exit(1)

    comparator = ModelComparator(
        baseline=baseline,
        executorch_models=list(executorch),
        runs=runs,
        warmup=warmup,
        verbose=verbose,
    )

    try:
        comparator.compare()

        if save:
            comparator.save_results(save)
            click.echo(f"\n✓ Results saved to: {save}")

        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="./outputs", help="Output directory")
@click.option(
    "--backends",
    multiple=True,
    type=click.Choice(["portable", "xnnpack", "vulkan"]),
    help="Backends to test (default: all)",
)
@click.option("-r", "--runs", default=50, help="Benchmark runs per backend")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def optimize(model_path, output_dir, backends, runs, verbose):
    """Auto-optimize model by testing all backends."""
    from pyroclast.core import AutoOptimizer

    optimizer = AutoOptimizer(
        model_path=model_path,
        output_dir=output_dir,
        backends=list(backends) if backends else None,
        runs=runs,
        verbose=verbose,
    )

    try:
        winner = optimizer.optimize()

        if winner:
            click.echo(f"\n🏆 Winner: {winner.backend}")
            click.echo(f"   Latency: {winner.mean_latency_ms:.2f}ms")
            sys.exit(0)
        else:
            click.echo("✗ No successful backend found", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="./deployment", help="Output directory")
@click.option(
    "-t",
    "--template",
    type=click.Choice(["fastapi", "docker", "lambda", "streamlit"]),
    default="fastapi",
    help="Deployment template",
)
@click.option("-p", "--port", default=8000, help="Server port (for FastAPI/Docker)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def deploy(model_path, output_dir, template, port, verbose):
    """Generate production-ready deployment code."""
    from pyroclast.core import DeploymentGenerator

    generator = DeploymentGenerator(
        model_path=model_path,
        output_dir=output_dir,
        template=template,
        port=port,
        verbose=verbose,
    )

    try:
        output_path = generator.generate()
        click.echo(f"\n✓ Deployment code generated!")
        click.echo(f"   Location: {output_path}")
        click.echo(f"   Template: {template}")
        click.echo(f"\nCheck README.md in the output directory for instructions.")
        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command(name="batch-test")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Save results to JSON file")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def batch_test(model_path, test_file, output, verbose):
    """Run batch tests from JSON file."""
    from pyroclast.core import BatchTester

    tester = BatchTester(
        model_path=model_path,
        test_file=test_file,
        output_file=output,
        verbose=verbose,
    )

    exit_code = tester.run()
    sys.exit(exit_code)


# JIT commands group
@cli.group()
def jit():
    """TorchScript JIT compilation commands."""
    pass


@jit.command(name="convert")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="./outputs", help="Output directory")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["script", "trace"]),
    default="trace",
    help="Conversion method",
)
@click.option("--no-optimize", is_flag=True, help="Disable optimization")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def jit_convert(model_path, output_dir, method, no_optimize, verbose):
    """Convert model to TorchScript format."""
    from pyroclast.core.jit.converter import JITConverter

    converter = JITConverter(
        model_path=model_path,
        output_dir=output_dir,
        method=method,
        optimize=not no_optimize,
        verbose=verbose,
    )

    try:
        output_path = converter.convert()
        click.echo(f"\n✓ JIT model saved to: {output_path}")
        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@jit.command(name="run")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "-d",
    "--device",
    type=click.Choice(["cpu", "cuda", "vulkan"]),
    default="cpu",
    help="Execution device",
)
@click.option("-b", "--benchmark", is_flag=True, help="Run benchmark")
@click.option("-r", "--runs", default=100, help="Number of benchmark runs")
@click.option("-w", "--warmup", default=10, help="Number of warmup runs")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def jit_run(model_path, device, benchmark, runs, warmup, verbose):
    """Run TorchScript model."""
    from pyroclast.core.jit.runner import JITRunner

    runner = JITRunner(model_path, device=device, verbose=verbose)

    try:
        if benchmark:
            stats = runner.benchmark(runs=runs, warmup=warmup)
            click.echo("\nBenchmark Results:")
            click.echo(f"  Mean:       {stats['mean_ms']:.2f}ms")
            click.echo(f"  Throughput: {stats['throughput']:.2f}/sec")
        else:
            output = runner.run()
            click.echo(f"✓ Inference completed")

        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
