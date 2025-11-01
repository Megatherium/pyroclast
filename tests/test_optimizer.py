"""
Tests for etorch_optimizer module
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from etorch_optimizer import BackendResult, AutoOptimizer


class TestBackendResult:
    """Tests for BackendResult class"""

    @pytest.mark.unit
    def test_init(self):
        """Test BackendResult initialization"""
        result = BackendResult("xnnpack")

        assert result.backend == "xnnpack"
        assert result.conversion_success == False
        assert result.conversion_time == 0.0
        assert result.conversion_error is None
        assert result.model_path is None
        assert result.model_size_mb == 0.0
        assert result.benchmark_success == False
        assert result.mean_latency_ms == 0.0

    @pytest.mark.unit
    def test_to_dict(self):
        """Test BackendResult to_dict conversion"""
        result = BackendResult("portable")
        result.conversion_success = True
        result.conversion_time = 1.5
        result.model_size_mb = 13.5
        result.mean_latency_ms = 150.0
        result.benchmark_success = True

        data = result.to_dict()

        assert data["backend"] == "portable"
        assert data["conversion"]["success"] == True
        assert data["conversion"]["time_seconds"] == 1.5
        assert data["benchmark"]["mean_latency_ms"] == 150.0


class TestAutoOptimizer:
    """Tests for AutoOptimizer class"""

    @pytest.mark.unit
    def test_init_default_backends(self, temp_dir):
        """Test AutoOptimizer initialization with default backends"""
        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir),
            runs=10
        )

        assert optimizer.model_path == Path("models/test")
        assert optimizer.output_dir == temp_dir
        assert optimizer.runs == 10
        assert "portable" in optimizer.backends
        assert "xnnpack" in optimizer.backends

    @pytest.mark.unit
    def test_init_custom_backends(self, temp_dir):
        """Test AutoOptimizer initialization with custom backends"""
        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir),
            backends=["portable"]
        )

        assert optimizer.backends == ["portable"]

    @pytest.mark.integration
    @patch('etorch_optimizer.ExecutorchConverter')
    @patch('etorch_runner.ExecutorchRunner')
    def test_test_backend_success(self, mock_runner_class, mock_converter_class, temp_dir):
        """Test successful backend testing"""
        # Setup mocks
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.return_value = temp_dir / "test.pte"

        # Create actual file for size calculation
        test_model = temp_dir / "test.pte"
        test_model.write_bytes(b"fake model data" * 1000)
        mock_converter.convert.return_value = test_model

        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner

        mock_stats = Mock()
        mock_stats.summary.return_value = {
            "mean_ms": 100.0,
            "median_ms": 95.0,
            "min_ms": 80.0,
            "max_ms": 120.0,
            "p95_ms": 110.0,
            "p99_ms": 115.0,
            "throughput": 10.0
        }
        mock_runner.benchmark.return_value = mock_stats

        # Test
        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir),
            runs=10,
            verbose=False
        )

        result = optimizer.test_backend("xnnpack")

        # Verify
        assert result.backend == "xnnpack"
        assert result.conversion_success == True
        assert result.benchmark_success == True
        assert result.mean_latency_ms == 100.0
        assert result.throughput == 10.0
        assert result.model_size_mb > 0

    @pytest.mark.integration
    @patch('etorch_optimizer.ExecutorchConverter')
    def test_test_backend_conversion_failure(self, mock_converter_class, temp_dir):
        """Test backend testing with conversion failure"""
        # Setup mock to fail
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.side_effect = Exception("Conversion failed")

        # Test
        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir),
            runs=10
        )

        result = optimizer.test_backend("vulkan")

        # Verify
        assert result.backend == "vulkan"
        assert result.conversion_success == False
        assert result.conversion_error == "Conversion failed"
        assert result.benchmark_success == False

    @pytest.mark.integration
    @patch('etorch_runner.ExecutorchRunner')
    @patch('etorch_optimizer.ExecutorchConverter')
    def test_test_backend_benchmark_failure(self, mock_converter_class, mock_runner_class, temp_dir):
        """Test backend testing with benchmark failure"""
        # Setup conversion to succeed
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        test_model = temp_dir / "test.pte"
        test_model.write_bytes(b"fake")
        mock_converter.convert.return_value = test_model

        # Setup benchmark to fail
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        mock_runner.benchmark.side_effect = Exception("Benchmark failed")

        # Test
        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir),
            runs=10
        )

        result = optimizer.test_backend("portable")

        # Verify
        assert result.conversion_success == True
        assert result.benchmark_success == False
        assert result.benchmark_error == "Benchmark failed"

    @pytest.mark.integration
    def test_analyze_results_no_success(self, temp_dir):
        """Test analyze_results with no successful backends"""
        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir)
        )

        # Create failed results
        result1 = BackendResult("portable")
        result1.conversion_success = False

        result2 = BackendResult("xnnpack")
        result2.conversion_success = True
        result2.benchmark_success = False

        optimizer.results = [result1, result2]

        winner = optimizer.analyze_results()
        assert winner is None

    @pytest.mark.integration
    def test_analyze_results_single_success(self, temp_dir):
        """Test analyze_results with single successful backend"""
        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir)
        )

        # Create one successful result
        result = BackendResult("portable")
        result.conversion_success = True
        result.benchmark_success = True
        result.mean_latency_ms = 150.0

        optimizer.results = [result]

        winner = optimizer.analyze_results()
        assert winner.backend == "portable"
        assert winner.mean_latency_ms == 150.0

    @pytest.mark.integration
    def test_analyze_results_multiple_success(self, temp_dir):
        """Test analyze_results picks fastest backend"""
        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir)
        )

        # Create multiple successful results
        result1 = BackendResult("portable")
        result1.conversion_success = True
        result1.benchmark_success = True
        result1.mean_latency_ms = 1500.0  # Slow

        result2 = BackendResult("xnnpack")
        result2.conversion_success = True
        result2.benchmark_success = True
        result2.mean_latency_ms = 85.0  # Fast

        result3 = BackendResult("vulkan")
        result3.conversion_success = True
        result3.benchmark_success = True
        result3.mean_latency_ms = 100.0  # Medium

        optimizer.results = [result1, result2, result3]

        winner = optimizer.analyze_results()
        assert winner.backend == "xnnpack"  # Fastest
        assert winner.mean_latency_ms == 85.0

    @pytest.mark.integration
    def test_save_results(self, temp_dir):
        """Test saving results to JSON"""
        import json

        optimizer = AutoOptimizer(
            model_path="models/test",
            output_dir=str(temp_dir)
        )

        result = BackendResult("xnnpack")
        result.conversion_success = True
        result.mean_latency_ms = 100.0

        optimizer.results = [result]

        output_file = optimizer.save_results(result)

        # Verify file exists and has correct content
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        assert data["winner"] == "xnnpack"
        assert data["model_path"] == "models/test"
        assert len(data["results"]) == 1
