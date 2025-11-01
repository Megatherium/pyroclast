"""
Utility functions for the Executorch toolkit.
"""

import sys


class Colors:
    """ANSI colors"""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def print_header(text, width=80):
    """Prints a centered header with a border."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*width}{Colors.END}\n")


def print_subheader(text):
    """Prints a subheader with a separator."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}▶ {text}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.END}")


def print_success(text):
    """Prints a success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_warning(text):
    """Prints a warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def print_error(text):
    """Prints an error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}", file=sys.stderr)


def print_info(key, value, unit=""):
    """Prints a key-value pair."""
    print(f"  {Colors.BLUE}{key}:{Colors.END} {Colors.BOLD}{value}{Colors.END} {unit}")


def print_dim(text):
    """Prints a dim message."""
    print(f"{Colors.DIM}{text}{Colors.END}")


def print_step(text):
    """Prints a step message."""
    print(f"{Colors.BLUE}▶{Colors.END} {text}")


def format_number(n):
    """Formats a number with commas."""
    return f"{n:,}"


def format_bytes(b):
    """Formats bytes into KB, MB, GB, etc."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.2f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.2f} MB"
    else:
        return f"{b/1024**3:.2f} GB"


def format_latency(ms):
    """Formats latency in milliseconds."""
    if ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"


# Input file handling utilities
import json
from pathlib import Path
from typing import Dict, List, Union, Any


class InputSpec:
    """Represents a single input specification"""

    def __init__(
        self, description: str = None, prompt: str = None, metadata: Dict[str, Any] = None
    ):
        self.description = description
        self.prompt = prompt
        self.metadata = metadata or {}

    def __repr__(self):
        return f"InputSpec(description={self.description!r}, prompt={self.prompt!r})"


def load_input_file(file_path: str) -> Union[InputSpec, List[InputSpec]]:
    """
    Load input specifications from a JSON file.

    Supports two formats:
    1. Single input:
       {
         "description": "...",
         "prompt": "...",
         "metadata": {...}
       }

    2. Batch inputs:
       {
         "inputs": [
           {"description": "...", "prompt": "..."},
           {"description": "...", "prompt": "..."}
         ]
       }

    Args:
        file_path: Path to JSON input file

    Returns:
        Single InputSpec or list of InputSpec objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

    # Check if it's a batch format
    if "inputs" in data:
        if not isinstance(data["inputs"], list):
            raise ValueError("'inputs' field must be a list")

        specs = []
        for i, item in enumerate(data["inputs"]):
            if not isinstance(item, dict):
                raise ValueError(f"Input {i} must be a dictionary")
            specs.append(
                InputSpec(
                    description=item.get("description"),
                    prompt=item.get("prompt"),
                    metadata=item.get("metadata", {}),
                )
            )
        return specs

    # Single input format
    return InputSpec(
        description=data.get("description"),
        prompt=data.get("prompt"),
        metadata=data.get("metadata", {}),
    )


def validate_input_spec(
    spec: InputSpec, require_description: bool = False, require_prompt: bool = False
) -> None:
    """
    Validate an input specification.

    Args:
        spec: InputSpec to validate
        require_description: Whether description is required
        require_prompt: Whether prompt is required

    Raises:
        ValueError: If validation fails
    """
    if require_description and not spec.description:
        raise ValueError("Input specification requires 'description' field")

    if require_prompt and not spec.prompt:
        raise ValueError("Input specification requires 'prompt' field")
