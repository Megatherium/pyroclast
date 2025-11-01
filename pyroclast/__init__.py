"""
Pyroclast - Executorch Toolkit

A comprehensive toolkit for converting, running, analyzing, and benchmarking
PyTorch models with Executorch.
"""

__version__ = "2.0.0"
__author__ = "Pyroclast Contributors"

# Re-export main CLI
from .cli import cli, main

# Re-export core classes
from .core import (
    ExecutorchConverter,
    ExecutorchRunner,
    ModelAnalyzer,
    ModelComparator,
    AutoOptimizer,
    ModelDoctor,
)

# Re-export utilities
from .utils import Colors, print_success, print_error, print_warning

__all__ = [
    # CLI
    "cli",
    "main",
    # Core
    "ExecutorchConverter",
    "ExecutorchRunner",
    "ModelAnalyzer",
    "ModelComparator",
    "AutoOptimizer",
    "ModelDoctor",
    # Utils
    "Colors",
    "print_success",
    "print_error",
    "print_warning",
]
