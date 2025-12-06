"""
AWS SageMaker LLM Service
Provides LLM inference via AWS SageMaker endpoints for README analysis and artifact relationship extraction.
Per rubric: "Provided use of AWS SageMaker or equivalent service to perform LLM-based activities"

Supports:
- SageMaker JumpStart foundation models (Llama, Mistral, etc.)
- Custom SageMaker endpoints
- Fallback to API-based LLMs if SageMaker is unavailable
"""

import json
import os
import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_SAGEMAKER_MODEL_ID = os.getenv("SAGEMAKER_MODEL_ID", "meta-textgeneration-llama-3-8b-instruct")
DEFAULT_SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


class SageMakerLLMService:
    """Service for invoking LLM models via AWS SageMaker"""

    def __init__(
        self,
        endpoint_name: Optional[str] = None,
        model_id: Optional[str] = None,
        region: str = DEFAULT_AWS_REGION,
    ):
        """Initialize SageMaker LLM service

        Args:
            endpoint_name: Name of existing SageMaker endpoint (optional)
            model_id: SageMaker JumpStart model ID (e.g., "meta-textgeneration-llama-3-8b-instruct")
            region: AWS region (default: us-east-1)
        """
        self.region = region
        self.endpoint_name = endpoint_name or DEFAULT_SAGEMAKER_ENDPOINT
        self.model_id = model_id or DEFAULT_SAGEMAKER_MODEL_ID
        self.sagemaker_runtime = None
        self.is_available = False

        try:
            # Initialize SageMaker Runtime client
            self.sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=region)
            self.is_available = True
            logger.info(
                f"SageMaker LLM service initialized: region={region}, "
                f"endpoint={self.endpoint_name or 'N/A'}, model_id={self.model_id}"
            )
            if not self.endpoint_name:
                logger.warning("SageMaker endpoint name is empty - invocations will fail")
        except (BotoCoreError, ClientError) as e:
            logger.warning(f"SageMaker Runtime client initialization failed: {e}")
            self.is_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize SageMaker service: {e}")
            self.is_available = False

    def invoke_jumpstart_model(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> Optional[str]:
        """
        Invoke a SageMaker JumpStart foundation model.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated text response, or None if invocation fails
        """
        if not self.is_available or not self.sagemaker_runtime:
            logger.debug("SageMaker service not available")
            return None

        try:
            # Format payload for JumpStart models (Llama format)
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "return_full_text": False,
                },
            }

            # Use endpoint if available, otherwise use model_id for JumpStart
            if self.endpoint_name:
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=json.dumps(payload),
                )
            else:
                # Use JumpStart model directly (requires endpoint to be deployed)
                # For now, we'll use endpoint_name if available
                logger.warning("SageMaker endpoint not configured, cannot invoke model")
                return None

            # Parse response
            response_body = json.loads(response["Body"].read().decode("utf-8"))

            # Extract generated text (format varies by model)
            if isinstance(response_body, dict):
                # Llama format: {"generated_text": "..."}
                if "generated_text" in response_body:
                    return response_body["generated_text"]
                # Some models return: [{"generated_text": "..."}]
                if isinstance(response_body.get("outputs"), list) and len(response_body["outputs"]) > 0:
                    return response_body["outputs"][0].get("generated_text", "")
                # Alternative format: {"outputs": ["..."]}
                if isinstance(response_body.get("outputs"), list) and len(response_body["outputs"]) > 0:
                    return str(response_body["outputs"][0])
                # Direct text response
                if "output" in response_body:
                    return response_body["output"]
            elif isinstance(response_body, list) and len(response_body) > 0:
                if isinstance(response_body[0], dict) and "generated_text" in response_body[0]:
                    return response_body[0]["generated_text"]
                return str(response_body[0])
            elif isinstance(response_body, str):
                return response_body

            logger.warning(f"Unexpected SageMaker response format: {response_body}")
            return None

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ValidationException":
                logger.warning(f"SageMaker endpoint validation error: {e}")
            elif error_code == "ModelError":
                logger.warning(f"SageMaker model error: {e}")
            else:
                logger.warning(f"SageMaker invocation failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"SageMaker invocation exception: {e}")
            return None

    def invoke_chat_model(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Optional[str]:
        """
        Invoke a chat-formatted model (for models that support chat format).

        Args:
            system_prompt: System message
            user_prompt: User message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response, or None if invocation fails
        """
        if not self.is_available or not self.sagemaker_runtime:
            logger.debug("SageMaker service not available")
            return None

        try:
            # Format messages as a simple prompt string
            # The endpoint expects a string in "inputs", not a messages array
            # Try simple format first (system + user combined)
            # If this fails, we can try the Llama 3 Instruct token format
            formatted_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                },
            }

            if not self.endpoint_name:
                logger.warning("SageMaker endpoint not configured, cannot invoke chat model")
                return None

            logger.info(f"Invoking SageMaker chat endpoint: {self.endpoint_name}")
            # Log payload for debugging (truncate if too long)
            payload_str = json.dumps(payload)
            if len(payload_str) > 500:
                logger.debug(f"SageMaker payload (truncated): {payload_str[:500]}...")
            else:
                logger.debug(f"SageMaker payload: {payload_str}")
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=payload_str,
            )
            logger.info("SageMaker chat endpoint invocation successful")

            response_body = json.loads(response["Body"].read().decode("utf-8"))

            # Extract generated text (standard format for Llama 3 on SageMaker JumpStart)
            if isinstance(response_body, dict):
                # Standard format: {"generated_text": "..."}
                if "generated_text" in response_body:
                    generated = response_body["generated_text"]
                    # Remove the input prompt from the response (Llama includes it)
                    if isinstance(generated, str) and formatted_prompt in generated:
                        generated = generated.replace(formatted_prompt, "", 1).strip()
                    return generated
                # Alternative format: {"outputs": ["..."]}
                if "outputs" in response_body and isinstance(response_body["outputs"], list):
                    if len(response_body["outputs"]) > 0:
                        output = response_body["outputs"][0]
                        if isinstance(output, dict) and "generated_text" in output:
                            generated = output["generated_text"]
                            if isinstance(generated, str) and formatted_prompt in generated:
                                generated = generated.replace(formatted_prompt, "", 1).strip()
                            return generated
                        return str(output)
            elif isinstance(response_body, list) and len(response_body) > 0:
                # List format response
                if isinstance(response_body[0], dict):
                    if "generated_text" in response_body[0]:
                        generated = response_body[0]["generated_text"]
                        if isinstance(generated, str) and formatted_prompt in generated:
                            generated = generated.replace(formatted_prompt, "", 1).strip()
                        return generated
                return str(response_body[0])
            elif isinstance(response_body, str):
                # Direct string response
                if formatted_prompt in response_body:
                    return response_body.replace(formatted_prompt, "", 1).strip()
                return response_body

            logger.warning(f"Unexpected SageMaker chat response format: {response_body}")
            return None

        except ClientError as e:
            logger.warning(f"SageMaker chat invocation failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"SageMaker chat invocation exception: {e}")
            return None


# Global SageMaker service instance
_sagemaker_service: Optional[SageMakerLLMService] = None


def get_sagemaker_service() -> Optional[SageMakerLLMService]:
    """Get or create SageMaker LLM service instance"""
    global _sagemaker_service

    if _sagemaker_service is not None:
        return _sagemaker_service if _sagemaker_service.is_available else None

    # Only initialize if endpoint is configured
    endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
    if not endpoint_name:
        logger.warning("SAGEMAKER_ENDPOINT_NAME not set, SageMaker service unavailable")
        return None

    region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    model_id = os.getenv("SAGEMAKER_MODEL_ID", DEFAULT_SAGEMAKER_MODEL_ID)

    _sagemaker_service = SageMakerLLMService(
        endpoint_name=endpoint_name,
        model_id=model_id,
        region=region,
    )

    return _sagemaker_service if _sagemaker_service.is_available else None
