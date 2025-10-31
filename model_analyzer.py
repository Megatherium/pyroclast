#!/usr/bin/env python3
"""
Model Analyzer - Deep inspection and analysis of PyTorch/Executorch models
Provides visualization, profiling, and architectural insights
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import torch
import torch.nn as nn


class Colors:
    """ANSI colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"


def format_bytes(bytes: int) -> str:
    """Format bytes into human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def draw_box(title: str, width: int = 70):
    """Draw a fancy box header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}╔{'═' * (width-2)}╗{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}║{title.center(width-2)}║{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}╚{'═' * (width-2)}╝{Colors.END}\n")


def print_tree_node(name: str, indent: int, is_last: bool, parent_last: List[bool]):
    """Print a tree node with proper indentation"""
    prefix = ""
    for i, last in enumerate(parent_last[:-1]):
        prefix += "    " if last else "│   "

    if indent > 0:
        connector = "└── " if is_last else "├── "
        prefix += connector

    print(f"{Colors.DIM}{prefix}{Colors.END}{Colors.CYAN}{name}{Colors.END}")


class ModelAnalyzer:
    """Comprehensive model analysis tool"""

    def __init__(self, model_path: str, verbose: bool = False):
        self.model_path = Path(model_path)
        self.verbose = verbose
        self.model = None
        self.analysis = {}

    def load_model(self) -> nn.Module:
        """Load PyTorch model"""
        draw_box("LOADING MODEL")

        print(f"{Colors.BLUE}▶ Path:{Colors.END} {self.model_path}")

        # Try different loading methods
        try:
            # Try HuggingFace first
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(str(self.model_path))
            print(f"{Colors.GREEN}✓{Colors.END} Loaded as HuggingFace model")
        except:
            try:
                # Try direct PyTorch load
                self.model = torch.load(self.model_path)
                print(f"{Colors.GREEN}✓{Colors.END} Loaded as PyTorch model")
            except Exception as e:
                print(f"{Colors.RED}✗ Failed to load: {e}{Colors.END}")
                raise

        self.model.eval()
        return self.model

    def count_parameters(self) -> Dict[str, Any]:
        """Count model parameters"""
        draw_box("PARAMETER ANALYSIS")

        total_params = 0
        trainable_params = 0
        frozen_params = 0

        param_by_layer = {}
        param_by_type = defaultdict(int)

        for name, param in self.model.named_parameters():
            count = param.numel()
            total_params += count

            if param.requires_grad:
                trainable_params += count
            else:
                frozen_params += count

            # Group by layer
            layer_name = name.split('.')[0] if '.' in name else name
            if layer_name not in param_by_layer:
                param_by_layer[layer_name] = 0
            param_by_layer[layer_name] += count

            # Group by type
            param_type = type(param).__name__
            param_by_type[param_type] += count

        memory_size = sum(p.numel() * p.element_size() for p in self.model.parameters())

        print(f"{Colors.CYAN}Total Parameters:{Colors.END} {Colors.BOLD}{format_number(total_params)}{Colors.END}")
        print(f"{Colors.CYAN}Trainable:{Colors.END} {format_number(trainable_params)} ({100*trainable_params/total_params:.1f}%)")
        print(f"{Colors.CYAN}Frozen:{Colors.END} {format_number(frozen_params)} ({100*frozen_params/total_params:.1f}%)")
        print(f"{Colors.CYAN}Memory Size:{Colors.END} {format_bytes(memory_size)}")

        # Show top layers by parameter count
        print(f"\n{Colors.BOLD}Top Layers by Parameter Count:{Colors.END}")
        sorted_layers = sorted(param_by_layer.items(), key=lambda x: x[1], reverse=True)[:10]
        for layer, count in sorted_layers:
            percent = 100 * count / total_params
            bar_length = int(percent / 2)
            bar = "█" * bar_length
            print(f"  {layer:30s} {bar:50s} {format_number(count):>15s} ({percent:5.1f}%)")

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "memory_bytes": memory_size,
            "by_layer": param_by_layer
        }

    def analyze_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture"""
        draw_box("ARCHITECTURE ANALYSIS")

        module_types = defaultdict(int)
        module_tree = {}
        total_layers = 0

        # Count module types
        for name, module in self.model.named_modules():
            if name == "":  # Skip root
                continue
            total_layers += 1
            module_type = type(module).__name__
            module_types[module_type] += 1

        print(f"{Colors.CYAN}Total Layers:{Colors.END} {format_number(total_layers)}")
        print(f"{Colors.CYAN}Unique Module Types:{Colors.END} {len(module_types)}")

        # Display module type distribution
        print(f"\n{Colors.BOLD}Module Type Distribution:{Colors.END}")
        sorted_types = sorted(module_types.items(), key=lambda x: x[1], reverse=True)[:15]

        max_count = max(count for _, count in sorted_types)
        for module_type, count in sorted_types:
            percent = 100 * count / total_layers
            bar_length = int((count / max_count) * 30)
            bar = "▓" * bar_length
            print(f"  {module_type:35s} {bar:30s} {count:>6d} ({percent:5.1f}%)")

        return {
            "total_layers": total_layers,
            "module_types": dict(module_types)
        }

    def visualize_tree(self, max_depth: int = 3):
        """Visualize model as a tree"""
        draw_box("MODEL STRUCTURE TREE")

        def print_module_tree(module, name="model", indent=0, parent_last=[]):
            if indent > max_depth:
                return

            # Print current module
            if indent == 0:
                print(f"{Colors.BOLD}{Colors.GREEN}{name}{Colors.END}")
            else:
                is_last = parent_last[-1] if parent_last else False
                print_tree_node(name, indent, is_last, parent_last)

            # Get children
            children = list(module.named_children())

            for i, (child_name, child_module) in enumerate(children):
                is_last_child = (i == len(children) - 1)
                new_parent_last = parent_last + [is_last_child]

                # Format child info
                child_type = type(child_module).__name__
                child_params = sum(p.numel() for p in child_module.parameters())

                display_name = f"{child_name} [{child_type}]"
                if child_params > 0:
                    display_name += f" ({format_number(child_params)} params)"

                print_module_tree(child_module, display_name, indent + 1, new_parent_last)

        print_module_tree(self.model)

    def analyze_computation(self, input_shape: Tuple = (1, 3, 224, 224)):
        """Analyze computational requirements (FLOPs estimation)"""
        draw_box("COMPUTATIONAL ANALYSIS")

        print(f"{Colors.YELLOW}⚠ Computational analysis for custom models is approximate{Colors.END}")
        print(f"{Colors.CYAN}Input Shape:{Colors.END} {input_shape}")

        # Count operations heuristically
        conv_layers = sum(1 for m in self.model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)))
        linear_layers = sum(1 for m in self.model.modules() if isinstance(m, nn.Linear))
        attention_layers = sum(1 for m in self.model.modules() if 'Attention' in type(m).__name__)

        print(f"\n{Colors.CYAN}Convolutional Layers:{Colors.END} {conv_layers}")
        print(f"{Colors.CYAN}Linear Layers:{Colors.END} {linear_layers}")
        print(f"{Colors.CYAN}Attention Layers:{Colors.END} {attention_layers}")

        # Rough FLOP estimate
        total_params = sum(p.numel() for p in self.model.parameters())
        estimated_flops = total_params * 2  # Rough approximation

        print(f"{Colors.CYAN}Estimated FLOPs:{Colors.END} ~{format_number(estimated_flops)}")

    def generate_report(self, output_path: Optional[Path] = None):
        """Generate comprehensive analysis report"""
        report = {
            "model_path": str(self.model_path),
            "parameters": self.analysis.get("parameters", {}),
            "architecture": self.analysis.get("architecture", {}),
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n{Colors.GREEN}✓ Report saved to {output_path}{Colors.END}")

        return report

    def analyze(self, max_tree_depth: int = 3, save_report: bool = False):
        """Run complete analysis"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'MODEL ANALYZER'.center(70)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.END}")

        self.load_model()

        self.analysis["parameters"] = self.count_parameters()
        self.analysis["architecture"] = self.analyze_architecture()

        self.visualize_tree(max_depth=max_tree_depth)
        self.analyze_computation()

        if save_report:
            report_path = self.model_path.parent / f"{self.model_path.name}_analysis.json"
            self.generate_report(report_path)

        print(f"\n{Colors.BOLD}{Colors.GREEN}Analysis Complete!{Colors.END}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch model architecture and parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "model_path",
        help="Path to model (HuggingFace directory or .pt file)"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum tree depth for visualization (default: 3)"
    )

    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save analysis report as JSON"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    try:
        analyzer = ModelAnalyzer(args.model_path, verbose=args.verbose)
        analyzer.analyze(max_tree_depth=args.max_depth, save_report=args.save_report)
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.RED}✗ Analysis failed: {e}{Colors.END}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
