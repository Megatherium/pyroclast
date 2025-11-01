#!/usr/bin/env python3
"""
Executorch Doctor - Model Health Check & Diagnostics Tool

Analyzes PyTorch models for Executorch compatibility issues and provides
actionable diagnostics and optimization suggestions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer

from pyroclast.utils import Colors, format_bytes


class Issue:
    """Represents a compatibility or performance issue"""

    SEVERITY_ERROR = "error"
    SEVERITY_WARNING = "warning"
    SEVERITY_INFO = "info"

    def __init__(
        self,
        severity: str,
        title: str,
        description: str,
        fix: Optional[str] = None,
        location: Optional[str] = None,
    ):
        self.severity = severity
        self.title = title
        self.description = description
        self.fix = fix
        self.location = location

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "fix": self.fix,
            "location": self.location,
        }


class Suggestion:
    """Represents an optimization or improvement suggestion"""

    def __init__(
        self, title: str, description: str, impact: str = "medium", command: Optional[str] = None
    ):
        self.title = title
        self.description = description
        self.impact = impact  # low, medium, high
        self.command = command

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "command": self.command,
        }


class ModelDoctor:
    """Diagnoses PyTorch models for Executorch compatibility"""

    def __init__(self, model_path: str, verbose: bool = False):
        self.model_path = Path(model_path)
        self.verbose = verbose
        self.model = None
        self.issues: List[Issue] = []
        self.suggestions: List[Suggestion] = []
        self.stats: Dict[str, Any] = {}

    def diagnose(self) -> Dict[str, Any]:
        """Run full diagnostic suite"""
        print(f"{Colors.CYAN}🔍 Executorch Model Doctor{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")

        # Load model
        if not self._load_model():
            return self._generate_report()

        # Run diagnostic checks
        self._check_model_size()
        self._check_dynamic_shapes()
        self._check_operators()
        self._check_backend_compatibility()
        self._generate_suggestions()

        return self._generate_report()

    def _load_model(self) -> bool:
        """Load the model from path"""
        print(f"{Colors.BLUE}▶ Loading model from: {self.model_path}{Colors.RESET}")

        try:
            # Try loading as HuggingFace model
            if self.model_path.is_dir():
                self.model = AutoModel.from_pretrained(str(self.model_path))
                self.stats["model_type"] = "HuggingFace"
                self.stats["architecture"] = self.model.__class__.__name__
            else:
                # Try loading as PyTorch checkpoint
                checkpoint = torch.load(self.model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.model = checkpoint["model"]
                else:
                    self.model = checkpoint
                self.stats["model_type"] = "PyTorch"

            # Set to eval mode
            self.model.eval()

            print(f"{Colors.GREEN}✓ Model loaded successfully{Colors.RESET}\n")
            return True

        except Exception as e:
            self.issues.append(
                Issue(
                    severity=Issue.SEVERITY_ERROR,
                    title="Model Loading Failed",
                    description=str(e),
                    fix="Ensure the path points to a valid PyTorch model or HuggingFace checkpoint",
                )
            )
            print(f"{Colors.RED}✗ Failed to load model: {e}{Colors.RESET}\n")
            return False

    def _check_model_size(self):
        """Check model size and memory footprint"""
        if self.model is None:
            return

        print(f"{Colors.BLUE}▶ Checking model size...{Colors.RESET}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Estimate memory (assuming fp32)
        memory_bytes = total_params * 4

        self.stats["total_parameters"] = total_params
        self.stats["trainable_parameters"] = trainable_params
        self.stats["memory_footprint_bytes"] = memory_bytes

        print(f"  Total parameters: {total_params:,}")
        print(f"  Memory footprint: {format_bytes(memory_bytes)}\n")

        # Check for mobile deployment constraints
        if memory_bytes > 500 * 1024 * 1024:  # 500MB
            self.issues.append(
                Issue(
                    severity=Issue.SEVERITY_WARNING,
                    title="Large Model Size",
                    description=f"Model size is {format_bytes(memory_bytes)}, which may exceed mobile device limits",
                    fix="Consider applying quantization to reduce size by ~4x",
                )
            )

    def _check_dynamic_shapes(self):
        """Check for dynamic shape usage"""
        if self.model is None:
            return

        print(f"{Colors.BLUE}▶ Checking for dynamic shapes...{Colors.RESET}")

        # This is a heuristic check - we look for common dynamic shape patterns
        dynamic_ops = []

        for name, module in self.model.named_modules():
            # Check for operations that commonly use dynamic shapes
            module_str = str(module)
            if any(keyword in module_str.lower() for keyword in ["adaptive", "dynamic"]):
                dynamic_ops.append(name)

        if dynamic_ops:
            self.issues.append(
                Issue(
                    severity=Issue.SEVERITY_WARNING,
                    title="Potential Dynamic Shapes Detected",
                    description=f"Found {len(dynamic_ops)} modules that may use dynamic shapes",
                    location=", ".join(dynamic_ops[:3]) + ("..." if len(dynamic_ops) > 3 else ""),
                    fix="Use fixed input shapes or enable dynamic shape support in Executorch",
                )
            )
            print(
                f"{Colors.YELLOW}  ⚠ Found {len(dynamic_ops)} potentially dynamic operations{Colors.RESET}\n"
            )
        else:
            print(f"{Colors.GREEN}  ✓ No obvious dynamic shape issues detected{Colors.RESET}\n")

    def _check_operators(self):
        """Check for unsupported operators"""
        if self.model is None:
            return

        print(f"{Colors.BLUE}▶ Checking operators...{Colors.RESET}")

        # Try to trace the model to find unsupported ops
        try:
            # Create a dummy input (this is a simplified check)
            # In practice, we'd need model-specific input shapes
            print(
                f"  {Colors.DIM}(Operator check requires model tracing - skipping for now){Colors.RESET}\n"
            )

            # Add informational issue about operator compatibility
            self.issues.append(
                Issue(
                    severity=Issue.SEVERITY_INFO,
                    title="Operator Compatibility Check",
                    description="Full operator compatibility check requires model tracing with sample inputs",
                    fix="Run actual conversion to identify unsupported operators",
                )
            )

        except Exception as e:
            if self.verbose:
                print(f"{Colors.YELLOW}  ⚠ Could not trace model: {e}{Colors.RESET}\n")

    def _check_backend_compatibility(self):
        """Check compatibility with different backends"""
        if self.model is None:
            return

        print(f"{Colors.BLUE}▶ Checking backend compatibility...{Colors.RESET}")

        # Check model architecture for backend-specific hints
        has_conv = any("conv" in name.lower() for name, _ in self.model.named_modules())
        has_attention = any("attention" in name.lower() for name, _ in self.model.named_modules())
        has_linear = any("linear" in name.lower() for name, _ in self.model.named_modules())

        backends = []

        # XNNPACK is good for conv and linear ops
        if has_conv or has_linear:
            backends.append("xnnpack (CPU-optimized)")

        # Portable is always compatible
        backends.append("portable (maximum compatibility)")

        # Vulkan for GPU acceleration
        backends.append("vulkan (GPU, if available)")

        self.stats["recommended_backends"] = backends

        print(f"  Recommended backends:")
        for backend in backends:
            print(f"    • {backend}")
        print()

    def _generate_suggestions(self):
        """Generate optimization suggestions based on findings"""

        # Suggest quantization for large models
        if self.stats.get("memory_footprint_bytes", 0) > 100 * 1024 * 1024:
            self.suggestions.append(
                Suggestion(
                    title="Apply Quantization",
                    description="Reduce model size by ~4x with minimal accuracy loss",
                    impact="high",
                    command="python3 pyroclast.py convert <model> --quantize",
                )
            )

        # Suggest auto-optimizer
        self.suggestions.append(
            Suggestion(
                title="Auto-Select Best Backend",
                description="Test all backends and automatically pick the fastest",
                impact="high",
                command="python3 pyroclast.py optimize <model> -o outputs/",
            )
        )

        # Suggest trying portable backend first
        if any(issue.severity == Issue.SEVERITY_ERROR for issue in self.issues):
            self.suggestions.append(
                Suggestion(
                    title="Try Portable Backend First",
                    description="Maximum compatibility for debugging conversion issues",
                    impact="medium",
                    command="python3 pyroclast.py convert <model> -o outputs/ --backend portable",
                )
            )

    def _generate_report(self) -> Dict[str, Any]:
        """Generate diagnostic report"""

        # Print issues
        errors = [i for i in self.issues if i.severity == Issue.SEVERITY_ERROR]
        warnings = [i for i in self.issues if i.severity == Issue.SEVERITY_WARNING]
        infos = [i for i in self.issues if i.severity == Issue.SEVERITY_INFO]

        if errors or warnings:
            print(f"{Colors.RED}{'─' * 70}{Colors.RESET}")
            print(
                f"{Colors.RED}Issues Found ({len(errors)} errors, {len(warnings)} warnings):{Colors.RESET}\n"
            )

            for i, issue in enumerate(errors + warnings, 1):
                icon = "❌" if issue.severity == Issue.SEVERITY_ERROR else "⚠️"
                color = Colors.RED if issue.severity == Issue.SEVERITY_ERROR else Colors.YELLOW

                print(f"{color}{icon} {issue.title}{Colors.RESET}")
                print(f"   {issue.description}")
                if issue.location:
                    print(f"   Location: {issue.location}")
                if issue.fix:
                    print(f"   {Colors.GREEN}→ Fix: {issue.fix}{Colors.RESET}")
                print()
        else:
            print(f"{Colors.GREEN}✓ No critical issues found!{Colors.RESET}\n")

        # Print suggestions
        if self.suggestions:
            print(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}")
            print(f"{Colors.CYAN}💡 Suggestions:{Colors.RESET}\n")

            for suggestion in self.suggestions:
                impact_color = {
                    "high": Colors.GREEN,
                    "medium": Colors.YELLOW,
                    "low": Colors.DIM,
                }.get(suggestion.impact, Colors.RESET)

                print(f"{impact_color}• {suggestion.title}{Colors.RESET}")
                print(f"  {suggestion.description}")
                if suggestion.command:
                    print(f"  {Colors.DIM}$ {suggestion.command}{Colors.RESET}")
                print()

        # Generate JSON report data
        report = {
            "model_path": str(self.model_path),
            "stats": self.stats,
            "issues": {
                "errors": [i.to_dict() for i in errors],
                "warnings": [i.to_dict() for i in warnings],
                "info": [i.to_dict() for i in infos],
            },
            "suggestions": [s.to_dict() for s in self.suggestions],
            "summary": {
                "total_issues": len(self.issues),
                "errors": len(errors),
                "warnings": len(warnings),
                "recommendations": len(self.suggestions),
            },
        }

        return report

    def save_report(self, output_path: str):
        """Save diagnostic report to JSON file"""
        report = self._generate_report()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"{Colors.GREEN}✓ Report saved to: {output_file}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Executorch Model Doctor - Diagnose compatibility issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic diagnosis
  python3 etorch_doctor.py models/my_model

  # With detailed output
  python3 etorch_doctor.py models/my_model -v

  # Save report to file
  python3 etorch_doctor.py models/my_model --save-report outputs/doctor_report.json
        """,
    )

    parser.add_argument("model_path", help="Path to PyTorch model or HuggingFace checkpoint")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--save-report", metavar="PATH", help="Save diagnostic report to JSON file")

    args = parser.parse_args()

    # Run diagnosis
    doctor = ModelDoctor(args.model_path, verbose=args.verbose)
    report = doctor.diagnose()

    # Save report if requested
    if args.save_report:
        doctor.save_report(args.save_report)

    # Exit with error code if critical issues found
    if report["summary"]["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
