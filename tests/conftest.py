"""
Pytest configuration and shared fixtures
"""
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_input_json(temp_dir):
    """Create a sample input JSON file"""
    input_file = temp_dir / "test_input.json"
    input_file.write_text("""{
  "description": "A test description",
  "prompt": "Test prompt text",
  "metadata": {"test_key": "test_value"}
}""")
    return input_file


@pytest.fixture
def sample_batch_json(temp_dir):
    """Create a sample batch input JSON file"""
    batch_file = temp_dir / "test_batch.json"
    batch_file.write_text("""{
  "inputs": [
    {"description": "First", "prompt": "First text"},
    {"description": "Second", "prompt": "Second text"}
  ]
}""")
    return batch_file
