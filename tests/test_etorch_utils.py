"""
Tests for etorch_utils module
"""
import json
import pytest
from pathlib import Path

from pyroclast.utils import (
    InputSpec,
    load_input_file,
    validate_input_spec,
    format_bytes,
    format_latency,
)


class TestInputSpec:
    """Tests for InputSpec class"""

    @pytest.mark.unit
    def test_init_with_defaults(self):
        """Test InputSpec initialization with defaults"""
        spec = InputSpec()
        assert spec.description is None
        assert spec.prompt is None
        assert spec.metadata == {}

    @pytest.mark.unit
    def test_init_with_values(self):
        """Test InputSpec initialization with values"""
        spec = InputSpec(
            description="Test desc",
            prompt="Test prompt",
            metadata={"key": "value"}
        )
        assert spec.description == "Test desc"
        assert spec.prompt == "Test prompt"
        assert spec.metadata == {"key": "value"}

    @pytest.mark.unit
    def test_repr(self):
        """Test InputSpec string representation"""
        spec = InputSpec(description="Desc", prompt="Prompt")
        repr_str = repr(spec)
        assert "InputSpec" in repr_str
        assert "Desc" in repr_str
        assert "Prompt" in repr_str


class TestLoadInputFile:
    """Tests for load_input_file function"""

    @pytest.mark.unit
    def test_load_single_input(self, sample_input_json):
        """Test loading single input format"""
        spec = load_input_file(str(sample_input_json))

        assert isinstance(spec, InputSpec)
        assert spec.description == "A test description"
        assert spec.prompt == "Test prompt text"
        assert spec.metadata == {"test_key": "test_value"}

    @pytest.mark.unit
    def test_load_batch_input(self, sample_batch_json):
        """Test loading batch input format"""
        specs = load_input_file(str(sample_batch_json))

        assert isinstance(specs, list)
        assert len(specs) == 2

        assert specs[0].description == "First"
        assert specs[0].prompt == "First text"

        assert specs[1].description == "Second"
        assert specs[1].prompt == "Second text"

    @pytest.mark.unit
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_input_file("nonexistent.json")

    @pytest.mark.unit
    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON raises ValueError"""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("not valid json {")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_input_file(str(invalid_file))

    @pytest.mark.unit
    def test_load_invalid_batch_format(self, temp_dir):
        """Test loading invalid batch format raises ValueError"""
        invalid_batch = temp_dir / "invalid_batch.json"
        invalid_batch.write_text('{"inputs": "not a list"}')

        with pytest.raises(ValueError, match="must be a list"):
            load_input_file(str(invalid_batch))

    @pytest.mark.unit
    def test_load_batch_with_non_dict_items(self, temp_dir):
        """Test loading batch with non-dict items raises ValueError"""
        invalid_batch = temp_dir / "invalid_items.json"
        invalid_batch.write_text('{"inputs": ["string", "not", "dicts"]}')

        with pytest.raises(ValueError, match="must be a dictionary"):
            load_input_file(str(invalid_batch))

    @pytest.mark.unit
    def test_load_minimal_input(self, temp_dir):
        """Test loading input with minimal fields"""
        minimal_file = temp_dir / "minimal.json"
        minimal_file.write_text('{}')

        spec = load_input_file(str(minimal_file))
        assert spec.description is None
        assert spec.prompt is None
        assert spec.metadata == {}


class TestValidateInputSpec:
    """Tests for validate_input_spec function"""

    @pytest.mark.unit
    def test_validate_no_requirements(self):
        """Test validation with no requirements passes"""
        spec = InputSpec()
        # Should not raise
        validate_input_spec(spec)

    @pytest.mark.unit
    def test_validate_require_description_success(self):
        """Test validation requiring description succeeds when present"""
        spec = InputSpec(description="Test")
        # Should not raise
        validate_input_spec(spec, require_description=True)

    @pytest.mark.unit
    def test_validate_require_description_failure(self):
        """Test validation requiring description fails when missing"""
        spec = InputSpec()
        with pytest.raises(ValueError, match="requires 'description'"):
            validate_input_spec(spec, require_description=True)

    @pytest.mark.unit
    def test_validate_require_prompt_success(self):
        """Test validation requiring prompt succeeds when present"""
        spec = InputSpec(prompt="Test")
        # Should not raise
        validate_input_spec(spec, require_prompt=True)

    @pytest.mark.unit
    def test_validate_require_prompt_failure(self):
        """Test validation requiring prompt fails when missing"""
        spec = InputSpec()
        with pytest.raises(ValueError, match="requires 'prompt'"):
            validate_input_spec(spec, require_prompt=True)

    @pytest.mark.unit
    def test_validate_both_requirements(self):
        """Test validation requiring both fields"""
        spec = InputSpec(description="Desc", prompt="Prompt")
        # Should not raise
        validate_input_spec(spec, require_description=True, require_prompt=True)


class TestFormatUtilities:
    """Tests for formatting utility functions"""

    @pytest.mark.unit
    def test_format_bytes_small(self):
        """Test formatting small byte values"""
        assert format_bytes(500) == "500 B"

    @pytest.mark.unit
    def test_format_bytes_kb(self):
        """Test formatting kilobyte values"""
        result = format_bytes(2048)
        assert "KB" in result
        assert "2.00" in result

    @pytest.mark.unit
    def test_format_bytes_mb(self):
        """Test formatting megabyte values"""
        result = format_bytes(2 * 1024 * 1024)
        assert "MB" in result
        assert "2.00" in result

    @pytest.mark.unit
    def test_format_bytes_gb(self):
        """Test formatting gigabyte values"""
        result = format_bytes(3 * 1024 * 1024 * 1024)
        assert "GB" in result
        assert "3.00" in result

    @pytest.mark.unit
    def test_format_latency_milliseconds(self):
        """Test formatting millisecond latencies"""
        result = format_latency(150.5)
        assert "ms" in result
        assert "150.50" in result

    @pytest.mark.unit
    def test_format_latency_seconds(self):
        """Test formatting second latencies"""
        result = format_latency(2500)
        assert "s" in result
        assert "2.50" in result
