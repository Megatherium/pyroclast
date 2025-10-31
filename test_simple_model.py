#!/usr/bin/env python3
"""
Test the converter with a simple model (MobileNetV2)
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from etorch_utils import (
    print_header,
    print_step,
    print_success,
    print_error,
    print_info,
    format_bytes,
)

try:
    from executorch.exir import to_edge, ExecutorchBackendConfig
    from torch.export import export
    EXEC_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    EXEC_OK = False


def test_simple_model():
    print_header("Testing Executorch with MobileNetV2", width=60)

    # Load a simple pre-trained model
    print_step("[1/5] Loading MobileNetV2...")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()
    print_success("Model loaded")

    # Create example input
    example_input = (torch.randn(1, 3, 224, 224),)
    print_step("[2/5] Creating example input...")
    print_info("Input shape", example_input[0].shape)

    # Export to EXIR
    print_step("[3/5] Exporting to EXIR...")
    with torch.no_grad():
        try:
            exported_program = export(model, example_input)
            print_success("Export successful")
        except Exception as e:
            print_error(f"Export failed: {e}")
            return False

    # Convert to Edge dialect
    print_step("[4/5] Converting to Edge dialect...")
    try:
        edge_program = to_edge(exported_program)
        print_success("Edge conversion successful")
    except Exception as e:
        print_error(f"Edge conversion failed: {e}")
        return False

    # Generate Executorch program
    print_step("[5/5] Generating Executorch program...")
    try:
        exec_program = edge_program.to_executorch()
        print_success("Executorch program generated")

        # Save
        output_path = Path("outputs/mobilenet_v2_test.pte")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(exec_program.buffer)

        file_size = len(exec_program.buffer)
        print_success(f"Saved to {output_path}")
        print_info("File size", format_bytes(file_size))

        return True

    except Exception as e:
        print_error(f"Executorch generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not EXEC_OK:
        print_error("Executorch imports failed!")
        sys.exit(1)

    success = test_simple_model()
    sys.exit(0 if success else 1)
