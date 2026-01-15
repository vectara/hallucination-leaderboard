"""Amazon Bedrock (Nova) model implementations for hallucination evaluation.

This module provides the LLM implementation for Amazon's Nova model family,
supporting API-based inference via AWS Bedrock Runtime.

Classes:
    AmazonConfig: Configuration model for Nova model settings.
    AmazonSummary: Output model for Nova summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes (currently unused).
    AmazonLLM: Main LLM class implementing AbstractLLM for Nova models.

Attributes:
    COMPANY: Provider identifier string ("amazon").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
import json
import boto3

COMPANY = "amazon"
"""str: Provider identifier used for API key lookup and model registration."""


class AmazonConfig(BasicLLMConfig):
    """Configuration model for Amazon Nova models.

    Extends BasicLLMConfig with Amazon-specific settings for model selection
    and AWS Bedrock API configuration.

    Attributes:
        company: Provider identifier, fixed to "amazon".
        model_name: Name of the Nova model variant to use. Includes pro, lite,
            and micro tiers with various versions.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference ("api" for Bedrock, or local).
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["amazon"] = "amazon"
    model_name: Literal[
        "nova-pro-v2",
        "nova-2-lite-v1:0",
        "nova-lite-v1:0",
        "nova-micro-v1:0",
        "nova-pro-v1:0",
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class AmazonSummary(BasicSummary):
    """Output model for Amazon Nova summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for AWS Bedrock API client inference.

    Defines how the model should be invoked when using the Bedrock Runtime API.

    Attributes:
        CHAT_DEFAULT: Use the Bedrock invoke_model API with chat message format.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Amazon Nova models only support API inference.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported AWS Bedrock modes.
# Each model maps endpoint types to ClientMode enum values indicating how to
# invoke the Bedrock Runtime API.
client_mode_group = {
    "nova-pro-v2": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "nova-2-lite-v1:0": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "nova-lite-v1:0": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "nova-micro-v1:0": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "nova-pro-v1:0": {
        "chat": ClientMode.CHAT_DEFAULT
    },
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Amazon Nova models do not support local execution.
local_mode_group = {}

class AmazonLLM(AbstractLLM):
    """LLM implementation for Amazon Nova models via AWS Bedrock.

    Provides text summarization using Amazon's Nova model family via the
    AWS Bedrock Runtime API. Supports chat-formatted generation with
    configurable model variants across pro, lite, and micro tiers.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        full_config: Complete configuration object for reference.
        model_fullname: Full Bedrock model ID (e.g., "us.amazon.nova-pro-v2").
    """

    def __init__(self, config: AmazonConfig):
        """Initialize the Amazon LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config
        self.model_fullname = f"us.amazon.{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Nova model via AWS Bedrock to generate a condensed
        summary. Strips surrounding quotes from the response if present.

        Args:
            prepared_text: The preprocessed text to summarize.

        Returns:
            The generated summary text, or an error placeholder if generation fails.

        Raises:
            Exception: If neither client nor local_model is initialized.
        """
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
                    response_package = self.client.invoke_model(
                        modelId=self.model_fullname,
                        body=json.dumps({
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"text": prepared_text}
                                    ]
                                }
                            ],
                            "inferenceConfig": {
                                "temperature": self.temperature,
                                "maxTokens": self.max_tokens
                            }
                        })
                    )
                    raw = response_package["body"].read()
                    model_response = json.loads(raw)

                    summary = model_response["output"]["message"]["content"][0]["text"]
                    summary = summary.strip()
                    if summary.startswith('"') and summary.endswith('"'):
                        summary = summary[1:-1].strip()
        elif self.local_model: 
            pass
        else:
            raise Exception(
                ModelInstantiationError.MISSING_SETUP.format(
                    class_name=self.__class__.__name__
                )
            )
        return summary

    def setup(self):
        """Initialize the AWS Bedrock client for inference.

        Creates a boto3 Bedrock Runtime client configured for the us-west-2
        region. Requires AWS credentials to be configured (AWS_SECRET_ACCESS_KEY
        environment variable or other boto3 credential methods).

        Raises:
            AssertionError: If the AWS_SECRET_ACCESS_KEY environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"AWS_SECRET_ACCESS_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"AWS_SECRET_ACCESS_KEY"
                )
                self.client = boto3.client("bedrock-runtime", region_name="us-west-2")
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )
        elif self.execution_mode == "local":
            pass

    def teardown(self):
        """Clean up resources after inference is complete.

        Releases any held resources from the client or local model.
        Currently a no-op as cleanup is handled automatically.
        """
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        """Close the AWS Bedrock client connection.

        Currently a no-op as the boto3 client does not require explicit cleanup.
        """
        pass