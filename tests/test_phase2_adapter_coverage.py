"""
Tests for phase2_adapter.py to increase code coverage.
Tests cover EvalContext creation, metric calculation, and error handling.
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.phase2_adapter import (
    create_eval_context_from_model_data,
    calculate_phase2_metrics,
    calculate_phase2_net_score,
    orchestrate_phase2_metrics,
)


class TestCreateEvalContext:
    """Test create_eval_context_from_model_data function"""

    def test_create_eval_context_huggingface_url(self):
        """Test EvalContext creation with HuggingFace URL"""
        model_data = {"url": "https://huggingface.co/test/model"}
        context = create_eval_context_from_model_data(model_data)

        assert context.url == "https://huggingface.co/test/model"
        assert context.category == "MODEL"

    def test_create_eval_context_github_url(self):
        """Test EvalContext creation with GitHub URL"""
        model_data = {"url": "https://github.com/user/repo"}
        context = create_eval_context_from_model_data(model_data)

        assert context.category == "CODE"

    def test_create_eval_context_dataset_url(self):
        """Test EvalContext creation with dataset URL"""
        model_data = {"url": "https://example.com/datasets/test"}
        context = create_eval_context_from_model_data(model_data)

        assert context.category == "DATASET"

    def test_create_eval_context_hf_data_string(self):
        """Test EvalContext creation with hf_data as JSON string"""
        model_data = {
            "url": "https://huggingface.co/test/model",
            "hf_data": json.dumps({"key": "value"}),
        }
        context = create_eval_context_from_model_data(model_data)

        assert len(context.hf_data) == 1
        assert context.hf_data[0]["key"] == "value"

    def test_create_eval_context_hf_data_string_list(self):
        """Test EvalContext creation with hf_data as JSON string list"""
        model_data = {
            "url": "https://huggingface.co/test/model",
            "hf_data": json.dumps([{"key1": "value1"}, {"key2": "value2"}]),
        }
        context = create_eval_context_from_model_data(model_data)

        assert len(context.hf_data) == 2

    def test_create_eval_context_hf_data_dict(self):
        """Test EvalContext creation with hf_data as dict"""
        model_data = {
            "url": "https://huggingface.co/test/model",
            "hf_data": {"key": "value"},
        }
        context = create_eval_context_from_model_data(model_data)

        assert len(context.hf_data) == 1
        assert context.hf_data[0]["key"] == "value"

    def test_create_eval_context_hf_data_list_of_dicts(self):
        """Test EvalContext creation with hf_data as list of dicts"""
        model_data = {
            "url": "https://huggingface.co/test/model",
            "hf_data": [{"key1": "value1"}, {"key2": "value2"}],
        }
        context = create_eval_context_from_model_data(model_data)

        assert len(context.hf_data) == 2

    def test_create_eval_context_hf_data_list_of_strings(self):
        """Test EvalContext creation with hf_data as list of JSON strings"""
        model_data = {
            "url": "https://huggingface.co/test/model",
            "hf_data": [json.dumps({"key1": "value1"}), json.dumps({"key2": "value2"})],
        }
        context = create_eval_context_from_model_data(model_data)

        assert len(context.hf_data) == 2

    def test_create_eval_context_hf_data_invalid_json(self):
        """Test EvalContext creation with invalid JSON in hf_data string"""
        model_data = {
            "url": "https://huggingface.co/test/model",
            "hf_data": "invalid json {",
        }
        context = create_eval_context_from_model_data(model_data)

        assert len(context.hf_data) == 0

    def test_create_eval_context_gh_data_variations(self):
        """Test EvalContext creation with various gh_data formats"""
        # Test gh_data as string
        model_data = {
            "url": "https://github.com/user/repo",
            "gh_data": json.dumps({"key": "value"}),
        }
        context = create_eval_context_from_model_data(model_data)
        assert len(context.gh_data) == 1

        # Test gh_data as dict
        model_data = {
            "url": "https://github.com/user/repo",
            "gh_data": {"key": "value"},
        }
        context = create_eval_context_from_model_data(model_data)
        assert len(context.gh_data) == 1

        # Test gh_data as list
        model_data = {
            "url": "https://github.com/user/repo",
            "gh_data": [{"key1": "value1"}, {"key2": "value2"}],
        }
        context = create_eval_context_from_model_data(model_data)
        assert len(context.gh_data) == 2

    def test_create_eval_context_exception_handling(self):
        """Test EvalContext creation handles exceptions gracefully"""
        # Create model_data that will cause an exception in the try block
        model_data = {"url": "https://test.com/model", "hf_data": "invalid"}
        # Patch the json import inside the function to raise exception
        import json as json_module

        with patch.object(json_module, "loads", side_effect=Exception("Test error")):
            context = create_eval_context_from_model_data(model_data)
            # Should return minimal EvalContext on exception
            assert context.url == "https://test.com/model"


class TestCalculatePhase2Metrics:
    """Test calculate_phase2_metrics function"""

    @pytest.mark.asyncio
    async def test_calculate_phase2_metrics_success(self):
        """Test successful metric calculation"""
        model_data = {
            "url": "https://huggingface.co/test/model",
            "hf_data": [{"key": "value"}],
        }

        with patch("src.metrics.phase2_adapter.get_all_metrics") as mock_get_metrics:
            mock_metric_fn = AsyncMock(return_value=0.75)
            mock_get_metrics.return_value = [("test_metric", mock_metric_fn)]

            metrics, latencies = await calculate_phase2_metrics(model_data)

            assert "test_metric" in metrics
            assert metrics["test_metric"] == 0.75
            assert "test_metric" in latencies
            assert latencies["test_metric"] > 0

    @pytest.mark.asyncio
    async def test_calculate_phase2_metrics_dict_return(self):
        """Test metric calculation with dict return value"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.get_all_metrics") as mock_get_metrics:
            mock_metric_fn = AsyncMock(return_value={"device1": 0.8, "device2": 0.6})
            mock_get_metrics.return_value = [("size_score", mock_metric_fn)]

            metrics, latencies = await calculate_phase2_metrics(model_data)

            assert "size_score" in metrics
            # Metric resilience may raise the floor, so ensure at least the max of the inputs.
            assert metrics["size_score"] >= 0.8

    @pytest.mark.asyncio
    async def test_calculate_phase2_metrics_dict_empty(self):
        """Test metric calculation with empty dict return"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.get_all_metrics") as mock_get_metrics:
            mock_metric_fn = AsyncMock(return_value={})
            mock_get_metrics.return_value = [("test_metric", mock_metric_fn)]

            metrics, latencies = await calculate_phase2_metrics(model_data)

            assert metrics["test_metric"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_phase2_metrics_dict_non_numeric(self):
        """Test metric calculation with dict containing non-numeric values"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.get_all_metrics") as mock_get_metrics:
            mock_metric_fn = AsyncMock(return_value={"key1": "string", "key2": 0.7})
            mock_get_metrics.return_value = [("test_metric", mock_metric_fn)]

            metrics, latencies = await calculate_phase2_metrics(model_data)

            assert metrics["test_metric"] == 0.7  # Should use max numeric value

    @pytest.mark.asyncio
    async def test_calculate_phase2_metrics_invalid_return_type(self):
        """Test metric calculation with invalid return type"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.get_all_metrics") as mock_get_metrics:
            mock_metric_fn = AsyncMock(return_value="invalid")
            mock_get_metrics.return_value = [("test_metric", mock_metric_fn)]

            metrics, latencies = await calculate_phase2_metrics(model_data)

            assert metrics["test_metric"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_phase2_metrics_exception(self):
        """Test metric calculation handles exceptions"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.get_all_metrics") as mock_get_metrics:
            mock_metric_fn = AsyncMock(side_effect=Exception("Test error"))
            mock_get_metrics.return_value = [("test_metric", mock_metric_fn)]

            metrics, latencies = await calculate_phase2_metrics(model_data)

            assert metrics["test_metric"] == 0.0
            assert latencies["test_metric"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_phase2_metrics_multiple_metrics(self):
        """Test metric calculation with multiple metrics"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.get_all_metrics") as mock_get_metrics:
            mock_metric1 = AsyncMock(return_value=0.8)
            mock_metric2 = AsyncMock(return_value=0.6)
            mock_get_metrics.return_value = [("metric1", mock_metric1), ("metric2", mock_metric2)]

            metrics, latencies = await calculate_phase2_metrics(model_data)

            assert len(metrics) == 2
            assert metrics["metric1"] == 0.8
            assert metrics["metric2"] == 0.6

    @pytest.mark.asyncio
    async def test_calculate_phase2_metrics_exception_overall(self):
        """Test calculate_phase2_metrics handles overall exceptions"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.create_eval_context_from_model_data", side_effect=Exception("Error")):
            metrics, latencies = await calculate_phase2_metrics(model_data)

            assert metrics == {}
            assert latencies == {}


class TestCalculatePhase2NetScore:
    """Test calculate_phase2_net_score function"""

    def test_calculate_phase2_net_score_success(self):
        """Test successful net score calculation"""
        metrics = {"metric1": 0.8, "metric2": 0.6}

        with patch("src.metrics.phase2_adapter.calculate_net_score") as mock_calculate:
            mock_calculate.return_value = 0.7

            net_score, latency = calculate_phase2_net_score(metrics)

            assert net_score == 0.7
            assert latency > 0

    def test_calculate_phase2_net_score_with_size_score_dict(self):
        """Test net score calculation with size_score as dict"""
        metrics = {"size_score": {"device1": 0.8, "device2": 0.6}, "metric2": 0.7}

        with patch("src.metrics.phase2_adapter.calculate_net_score") as mock_calculate:
            mock_calculate.return_value = 0.75

            net_score, latency = calculate_phase2_net_score(metrics)

            # Should process size_score dict to max value
            mock_calculate.assert_called_once()
            call_metrics = mock_calculate.call_args[0][0]
            assert isinstance(call_metrics["size_score"], float)

    def test_calculate_phase2_net_score_size_score_empty_dict(self):
        """Test net score calculation with empty size_score dict"""
        metrics = {"size_score": {}, "metric2": 0.7}

        with patch("src.metrics.phase2_adapter.calculate_net_score") as mock_calculate:
            mock_calculate.return_value = 0.7

            net_score, latency = calculate_phase2_net_score(metrics)

            call_metrics = mock_calculate.call_args[0][0]
            assert call_metrics["size_score"] == 0.0

    def test_calculate_phase2_net_score_exception(self):
        """Test net score calculation handles exceptions"""
        metrics = {"metric1": 0.8}

        with patch("src.metrics.phase2_adapter.calculate_net_score", side_effect=Exception("Error")):
            net_score, latency = calculate_phase2_net_score(metrics)

            assert net_score == 0.0
            assert latency == 0.0


class TestOrchestratePhase2Metrics:
    """Test orchestrate_phase2_metrics function"""

    @pytest.mark.asyncio
    async def test_orchestrate_phase2_metrics_success(self):
        """Test successful metric orchestration"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.calculate_phase2_metrics") as mock_calc:
            mock_calc.return_value = ({"metric1": 0.8}, {"metric1": 0.1})

            with patch("src.metrics.phase2_adapter.calculate_phase2_net_score") as mock_net:
                mock_net.return_value = (0.75, 0.05)

                result = await orchestrate_phase2_metrics(model_data)

                assert "net_score" in result
                assert "sub_scores" in result
                assert "confidence_intervals" in result
                assert "latency_ms" in result
                assert result["net_score"] == 0.75
                assert result["sub_scores"]["metric1"] == 0.8

    @pytest.mark.asyncio
    async def test_orchestrate_phase2_metrics_confidence_intervals(self):
        """Test orchestration includes confidence intervals"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.calculate_phase2_metrics") as mock_calc:
            mock_calc.return_value = ({}, {})

            with patch("src.metrics.phase2_adapter.calculate_phase2_net_score") as mock_net:
                mock_net.return_value = (0.75, 0.05)

                result = await orchestrate_phase2_metrics(model_data)

                assert "confidence_intervals" in result
                assert "net_score" in result["confidence_intervals"]
                assert result["confidence_intervals"]["net_score"]["lower"] >= 0.0
                assert result["confidence_intervals"]["net_score"]["upper"] <= 1.0

    @pytest.mark.asyncio
    async def test_orchestrate_phase2_metrics_exception(self):
        """Test orchestration handles exceptions"""
        model_data = {"url": "https://huggingface.co/test/model"}

        with patch("src.metrics.phase2_adapter.calculate_phase2_metrics", side_effect=Exception("Error")):
            result = await orchestrate_phase2_metrics(model_data)

            assert result["net_score"] == 0.0
            assert result["sub_scores"] == {}
            assert result["latency_ms"] == 0
