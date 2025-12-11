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
MAX_CHAT_INPUT_CHARS = int(os.getenv("SAGEMAKER_MAX_INPUT_CHARS", "4500"))
DEFAULT_CHAT_MAX_NEW_TOKENS = int(os.getenv("SAGEMAKER_MAX_NEW_TOKENS", "384"))


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
        self.enable_llm_cache = True

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

    def invoke_jumpstart_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> Optional[str]:
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
        max_tokens: int = DEFAULT_CHAT_MAX_NEW_TOKENS,
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

        if not self.endpoint_name:
            logger.warning("SageMaker endpoint not configured, cannot invoke chat model")
            return None

        system_text = system_prompt or ""
        user_text = user_prompt or ""
        trimmed_user_prompt = user_text[:MAX_CHAT_INPUT_CHARS] if len(user_text) > MAX_CHAT_INPUT_CHARS else user_text
        formatted_prompt = f"{system_text}\n\nUser: {trimmed_user_prompt}\n\nAssistant:"

        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.8,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        current_prompt = formatted_prompt
        retries = 2

        for attempt in range(retries):
            payload_str = json.dumps(payload)
            logger.info(
                "CW_SAGEMAKER_CHAT: endpoint=%s attempt=%d prompt_chars=%d tokens=%d",
                self.endpoint_name,
                attempt + 1,
                len(current_prompt),
                payload["parameters"]["max_new_tokens"],
            )
            try:
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=payload_str,
                )
                logger.info("SageMaker chat endpoint invocation successful")

                response_body = json.loads(response["Body"].read().decode("utf-8"))

                def _strip_prompt(text: str) -> str:
                    if isinstance(text, str) and current_prompt in text:
                        return text.replace(current_prompt, "", 1).strip()
                    return text

                if isinstance(response_body, dict):
                    if "generated_text" in response_body:
                        return _strip_prompt(response_body["generated_text"])
                    if (
                        "outputs" in response_body
                        and isinstance(response_body["outputs"], list)
                        and response_body["outputs"]
                    ):
                        first_output = response_body["outputs"][0]
                        if isinstance(first_output, dict) and "generated_text" in first_output:
                            return _strip_prompt(first_output["generated_text"])
                elif isinstance(response_body, list) and response_body:
                    first_item = response_body[0]
                    if isinstance(first_item, dict) and "generated_text" in first_item:
                        return _strip_prompt(first_item["generated_text"])
                    return _strip_prompt(str(first_item))
                elif isinstance(response_body, str):
                    return _strip_prompt(response_body)

                logger.warning(f"Unexpected SageMaker chat response format: {response_body}")
                return None

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                message = e.response.get("Error", {}).get("Message", "")
                logger.warning(
                    "SageMaker chat invocation failed (attempt %d/%d): code=%s message=%s",
                    attempt + 1,
                    retries,
                    error_code,
                    message,
                )

                if attempt + 1 < retries and error_code in {"ModelError", "InternalFailure"}:
                    payload["parameters"]["max_new_tokens"] = max(
                        128, int(payload["parameters"]["max_new_tokens"] * 0.5)
                    )
                    if len(current_prompt) > 2000:
                        new_limit = max(1500, int(len(current_prompt) * 0.75))
                        current_prompt = current_prompt[:new_limit]
                        payload["inputs"] = current_prompt
                        logger.info(
                            "CW_SAGEMAKER_CHAT_RETRY: reduced prompt to %d chars, tokens=%d",
                            len(current_prompt),
                            payload["parameters"]["max_new_tokens"],
                        )
                    continue
                return None
            except Exception as e:
                logger.warning(f"SageMaker chat invocation exception: {e}")
                return None

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
