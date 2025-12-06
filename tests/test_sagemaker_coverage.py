"""
Tests for AWS SageMaker LLM integration to increase code coverage.
Tests cover initialization, invocation, error handling, and fallback scenarios.
"""

import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.aws.sagemaker_llm import SageMakerLLMService, get_sagemaker_service


class TestSageMakerLLMService:
    """Test SageMaker LLM Service initialization and methods"""

    def test_init_success(self):
        """Test successful initialization with valid boto3 client"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            service = SageMakerLLMService(
                endpoint_name="test-endpoint",
                model_id="test-model",
                region="us-east-1",
            )

            assert service.is_available is True
            assert service.endpoint_name == "test-endpoint"
            assert service.model_id == "test-model"
            assert service.region == "us-east-1"
            assert service.sagemaker_runtime == mock_runtime

    def test_init_boto_core_error(self):
        """Test initialization failure with BotoCoreError"""
        with patch("boto3.client", side_effect=BotoCoreError()):
            service = SageMakerLLMService()
            assert service.is_available is False
            assert service.sagemaker_runtime is None

    def test_init_client_error(self):
        """Test initialization failure with ClientError"""
        with patch("boto3.client", side_effect=ClientError({}, "operation")):
            service = SageMakerLLMService()
            assert service.is_available is False

    def test_init_generic_exception(self):
        """Test initialization failure with generic exception"""
        with patch("boto3.client", side_effect=Exception("Unexpected error")):
            service = SageMakerLLMService()
            assert service.is_available is False

    def test_invoke_jumpstart_model_not_available(self):
        """Test invoke_jumpstart_model when service is not available"""
        service = SageMakerLLMService()
        service.is_available = False
        result = service.invoke_jumpstart_model("test prompt")
        assert result is None

    def test_invoke_jumpstart_model_no_endpoint(self):
        """Test invoke_jumpstart_model when endpoint is not configured"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            service = SageMakerLLMService(endpoint_name="")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")
            assert result is None

    def test_invoke_jumpstart_model_success_dict_format(self):
        """Test successful invocation with dict response format"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {
                "Body": Mock(read=Mock(return_value=json.dumps({"generated_text": "Test response"}).encode()))
            }
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result == "Test response"
            mock_runtime.invoke_endpoint.assert_called_once()

    def test_invoke_jumpstart_model_success_outputs_list(self):
        """Test successful invocation with outputs list format"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {
                "Body": Mock(
                    read=Mock(return_value=json.dumps({"outputs": [{"generated_text": "List response"}]}).encode())
                )
            }
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result == "List response"

    def test_invoke_jumpstart_model_success_outputs_string_list(self):
        """Test successful invocation with outputs as string list"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {
                "Body": Mock(
                    read=Mock(return_value=json.dumps({"outputs": [{"generated_text": "String response"}]}).encode())
                )
            }
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result == "String response"

    def test_invoke_jumpstart_model_success_output_key(self):
        """Test successful invocation with 'output' key"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {
                "Body": Mock(read=Mock(return_value=json.dumps({"output": "Output key response"}).encode()))
            }
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result == "Output key response"

    def test_invoke_jumpstart_model_success_list_response(self):
        """Test successful invocation with list response format"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {
                "Body": Mock(read=Mock(return_value=json.dumps([{"generated_text": "List format response"}]).encode()))
            }
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result == "List format response"

    def test_invoke_jumpstart_model_success_string_response(self):
        """Test successful invocation with string response"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {"Body": Mock(read=Mock(return_value=json.dumps("String response").encode()))}
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result == "String response"

    def test_invoke_jumpstart_model_unexpected_format(self):
        """Test invocation with unexpected response format"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {"Body": Mock(read=Mock(return_value=json.dumps({"unexpected": "format"}).encode()))}
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result is None

    def test_invoke_jumpstart_model_client_error_validation(self):
        """Test invocation with ValidationException"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid request"}}
            mock_runtime.invoke_endpoint.side_effect = ClientError(error_response, "invoke_endpoint")

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result is None

    def test_invoke_jumpstart_model_client_error_model_error(self):
        """Test invocation with ModelError"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            error_response = {"Error": {"Code": "ModelError", "Message": "Model error"}}
            mock_runtime.invoke_endpoint.side_effect = ClientError(error_response, "invoke_endpoint")

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result is None

    def test_invoke_jumpstart_model_client_error_other(self):
        """Test invocation with other ClientError"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            error_response = {"Error": {"Code": "OtherError", "Message": "Other error"}}
            mock_runtime.invoke_endpoint.side_effect = ClientError(error_response, "invoke_endpoint")

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result is None

    def test_invoke_jumpstart_model_generic_exception(self):
        """Test invocation with generic exception"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_runtime.invoke_endpoint.side_effect = Exception("Unexpected error")

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_jumpstart_model("test prompt")

            assert result is None

    def test_invoke_chat_model_not_available(self):
        """Test invoke_chat_model when service is not available"""
        service = SageMakerLLMService()
        service.is_available = False
        result = service.invoke_chat_model("system", "user")
        assert result is None

    def test_invoke_chat_model_no_endpoint(self):
        """Test invoke_chat_model when endpoint is not configured"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            service = SageMakerLLMService(endpoint_name="")
            service.is_available = True
            result = service.invoke_chat_model("system", "user")
            assert result is None

    def test_invoke_chat_model_success(self):
        """Test successful chat model invocation"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {
                "Body": Mock(read=Mock(return_value=json.dumps({"generated_text": "Chat response"}).encode()))
            }
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_chat_model("system prompt", "user prompt")

            assert result == "Chat response"
            call_args = mock_runtime.invoke_endpoint.call_args
            assert call_args[1]["EndpointName"] == "test-endpoint"
            body = json.loads(call_args[1]["Body"])
            # Check for string inputs format (required for Llama 3.1 8B Instruct on SageMaker JumpStart)
            assert "inputs" in body
            assert isinstance(body["inputs"], str)
            # Verify Llama 3 Instruct format tokens are present
            assert "<|begin_of_text|>" in body["inputs"]
            assert "<|start_header_id|>system<|end_header_id|>" in body["inputs"]
            assert "<|start_header_id|>user<|end_header_id|>" in body["inputs"]
            assert "<|start_header_id|>assistant<|end_header_id|>" in body["inputs"]
            assert "<|eot_id|>" in body["inputs"]
            # Verify prompts are included
            assert "system prompt" in body["inputs"]
            assert "user prompt" in body["inputs"]
            # Check parameters
            assert "parameters" in body
            assert body["parameters"]["max_new_tokens"] == 1024
            assert body["parameters"]["temperature"] == 0.1

    def test_invoke_chat_model_success_outputs_list(self):
        """Test chat model with outputs list format"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {
                "Body": Mock(
                    read=Mock(return_value=json.dumps({"outputs": [{"generated_text": "Outputs response"}]}).encode())
                )
            }
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_chat_model("system", "user")

            assert result == "Outputs response"

    def test_invoke_chat_model_success_list_response(self):
        """Test chat model with list response format"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {
                "Body": Mock(read=Mock(return_value=json.dumps([{"generated_text": "List chat response"}]).encode()))
            }
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_chat_model("system", "user")

            assert result == "List chat response"

    def test_invoke_chat_model_success_string_response(self):
        """Test chat model with string response"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {"Body": Mock(read=Mock(return_value=json.dumps("String chat").encode()))}
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_chat_model("system", "user")

            assert result == "String chat"

    def test_invoke_chat_model_unexpected_format(self):
        """Test chat model with unexpected format"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_response = {"Body": Mock(read=Mock(return_value=json.dumps({"unexpected": "format"}).encode()))}
            mock_runtime.invoke_endpoint.return_value = mock_response

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_chat_model("system", "user")

            assert result is None

    def test_invoke_chat_model_client_error(self):
        """Test chat model with ClientError"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            error_response = {"Error": {"Code": "ValidationException", "Message": "Error"}}
            mock_runtime.invoke_endpoint.side_effect = ClientError(error_response, "invoke_endpoint")

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_chat_model("system", "user")

            assert result is None

    def test_invoke_chat_model_generic_exception(self):
        """Test chat model with generic exception"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            mock_runtime.invoke_endpoint.side_effect = Exception("Unexpected")

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = True
            result = service.invoke_chat_model("system", "user")

            assert result is None


class TestGetSageMakerService:
    """Test get_sagemaker_service function"""

    def test_get_service_no_endpoint_env(self):
        """Test get_sagemaker_service when endpoint env var is not set"""
        with patch.dict(os.environ, {}, clear=True):
            # Clear the global service instance
            import src.aws.sagemaker_llm

            src.aws.sagemaker_llm._sagemaker_service = None
            result = get_sagemaker_service()
            assert result is None

    def test_get_service_with_endpoint_env(self):
        """Test get_sagemaker_service when endpoint env var is set"""
        with patch.dict(os.environ, {"SAGEMAKER_ENDPOINT_NAME": "test-endpoint"}, clear=False):
            with patch("boto3.client") as mock_client:
                mock_runtime = Mock()
                mock_client.return_value = mock_runtime

                import src.aws.sagemaker_llm

                src.aws.sagemaker_llm._sagemaker_service = None
                result = get_sagemaker_service()

                assert result is not None
                assert result.is_available is True

    def test_get_service_cached(self):
        """Test get_sagemaker_service returns cached instance"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            import src.aws.sagemaker_llm

            # Create a service instance
            service = SageMakerLLMService(endpoint_name="test-endpoint")
            src.aws.sagemaker_llm._sagemaker_service = service

            result = get_sagemaker_service()
            assert result == service

    def test_get_service_cached_not_available(self):
        """Test get_sagemaker_service returns None for cached unavailable service"""
        with patch("boto3.client") as mock_client:
            mock_runtime = Mock()
            mock_client.return_value = mock_runtime

            import src.aws.sagemaker_llm

            service = SageMakerLLMService(endpoint_name="test-endpoint")
            service.is_available = False
            src.aws.sagemaker_llm._sagemaker_service = service

            result = get_sagemaker_service()
            assert result is None
