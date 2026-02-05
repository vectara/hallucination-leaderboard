"""Microsoft (Phi, Orca, WizardLM) model implementations for hallucination evaluation.

This module provides the LLM implementation for Microsoft's model families,
supporting API-based inference via the Azure AI Inference SDK. Includes
support for Phi series (2, 3, 3.5, 4), Orca, and WizardLM models.

Classes:
    MicrosoftConfig: Configuration model for Microsoft model settings.
    MicrosoftSummary: Output model for Microsoft summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes (currently unused).
    MicrosoftLLM: Main LLM class implementing AbstractLLM for Microsoft models.

Attributes:
    COMPANY: Provider identifier string ("microsoft").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "microsoft"
"""str: Provider identifier used for model registration."""


class MicrosoftConfig(BasicLLMConfig):
    """Configuration model for Microsoft models.

    Extends BasicLLMConfig with Microsoft/Azure-specific settings for model
    selection and API configuration. Supports Phi series, Orca, and WizardLM
    models via Azure AI Inference.

    Attributes:
        company: Provider identifier, fixed to "microsoft".
        model_name: Name of the Microsoft model variant to use. Includes Phi
            series (2, 3, 3.5, 4), Orca-2, and WizardLM-2.
        model_key: Azure API key for authentication.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        azure_endpoint: Azure AI endpoint URL for the model deployment.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["microsoft"]
    model_name: Literal[
        "Orca-2-13b",
        "phi-2",
        "Phi-3-mini-4k-instruct",
        "Phi-3-mini-128k-instruct",
        "Phi-3.5-mini-instruct",
        "Phi-3.5-MoE-instruct",
        "phi-4",
        "WizardLM-2-8x22B",


        "Phi-4-mini-instruct",
        "Phi-4",
        "microsoft-phi-2",  # Resource not active
        "microsoft-Orca-2-13b"  # Resource not active
    ]
    model_key: str = "NoneGiven"
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    azure_endpoint: str = "NoneGiven"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["default"] = "default"

class MicrosoftSummary(BasicSummary):
    """Output model for Microsoft summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None
    api_type: Literal["default"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for Azure AI Inference API client.

    Defines how the model should be invoked when using the Azure AI SDK.

    Attributes:
        CHAT_DEFAULT: Use ChatCompletionsClient.complete for chat generation.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Microsoft models are accessed via Azure AI API.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values indicating how to
# invoke the Azure AI Inference API. Some resources may not be active.
client_mode_group = {
    "Phi-4-mini-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Phi-4": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "microsoft-phi-2": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "microsoft-Orca-2-13b": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Microsoft models are accessed via Azure AI API only.
local_mode_group = {}

class MicrosoftLLM(AbstractLLM):
    """LLM implementation for Microsoft models.

    Provides text summarization using Microsoft's model families (Phi, Orca,
    WizardLM) via the Azure AI Inference SDK. Uses ChatCompletionsClient
    for chat-based generation with configurable endpoints.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        azure_endpoint: Azure AI endpoint URL for the model deployment.
        model_key: Azure API key for authentication.
    """

    def __init__(self, config: MicrosoftConfig):
        """Initialize the Microsoft LLM with the given configuration.

        Args:
            config: Configuration object specifying model and Azure settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type
        self.azure_endpoint = config.azure_endpoint
        self.model_key = config.model_key

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Microsoft model via the Azure AI Inference API
        to generate a condensed summary. Sends the text as a UserMessage
        to the ChatCompletionsClient.

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
                    response = self.client.complete(
                        messages=[
                            UserMessage(content=prepared_text),
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        model=self.model_fullname
                    )

                    summary = response.choices[0].message.content
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
        """Initialize the Azure AI Inference client for Microsoft model inference.

        Creates a ChatCompletionsClient instance using the model_key and
        azure_endpoint from the configuration. Uses API version 2024-05-01-preview.

        Raises:
            AssertionError: If the model_key is None.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "default":
                    api_key = self.model_key
                    assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = ChatCompletionsClient(
                        endpoint=self.azure_endpoint,
                        credential=AzureKeyCredential(api_key),
                        api_version="2024-05-01-preview"
                    )
                else:
                    raise ValueError(f"Unknown api_type: {self.api_type}")

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
        """Close the Azure AI client connection.

        Currently a no-op as the ChatCompletionsClient does not require
        explicit cleanup.
        """
        pass