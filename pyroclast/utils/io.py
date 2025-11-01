"""
I/O utilities for Pyroclast - input file handling, specifications, etc.
"""

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
