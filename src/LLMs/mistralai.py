"""Mistral AI model implementations for hallucination evaluation.

This module provides the LLM implementation for Mistral AI's model family,
supporting API-based inference via the official Mistral SDK. Includes support
for Mistral, Mixtral, Ministral, Magistral, and Pixtral model series.

Classes:
    MistralAIConfig: Configuration model for Mistral AI model settings.
    MistralAISummary: Output model for Mistral AI summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    MistralAILLM: Main LLM class implementing AbstractLLM for Mistral AI models.

Attributes:
    COMPANY: Provider identifier string ("mistralai").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from mistralai import Mistral

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "mistralai"
"""str: Provider identifier used for API key lookup and model registration."""

class MistralAIConfig(BasicLLMConfig):
    """Configuration model for Mistral AI models.

    Extends BasicLLMConfig with Mistral AI-specific settings for model selection
    and API configuration. Supports the full range of Mistral models including
    Mistral, Mixtral (MoE), Ministral (small), Magistral, and Pixtral (vision).

    Attributes:
        company: Provider identifier, fixed to "mistralai".
        model_name: Name of the Mistral AI model variant to use. Includes base
            Mistral (7B), Mixtral MoE (8x7B, 8x22B), Ministral (3B, 8B, 14B),
            Magistral, and Pixtral vision models.
        execution_mode: Where to run inference, currently only "api" supported.
        date_code: Optional version/date identifier for the model.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["mistralai"] = "mistralai"
    model_name: Literal[
        "Ministral-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Mistral-Nemo-Instruct",
        "Mistral-Small-3.1-24b-instruct",
        "Mistral-Small-24B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x22B-Instruct-v0.1",
        "Pixtral-Large-Instruct",
        


        "magistral-medium", 
        "mistral-medium",
        "mistral-small",
        "mistral-large",
        "ministral-3b",
        "ministral-8b",
        "ministral-14b",
        "pixtral-large",
        "pixtral-12b",
        "open-mistral-nemo"
    ]
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"

class MistralAISummary(BasicSummary):
    """Output model for Mistral AI summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore" 

class ClientMode(Enum):
    """Execution modes for Mistral AI API client inference.

    Defines how the model should be invoked when using the official Mistral SDK.

    Attributes:
        CHAT_DEFAULT: Use chat.complete for conversational generation.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Mistral AI models are accessed via API only.

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
# invoke the Mistral AI API. All models use CHAT_DEFAULT for chat completion.
client_mode_group = {
    "magistral-medium": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "mistral-medium":{
        "chat": ClientMode.CHAT_DEFAULT
    },
    "mistral-small": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "mistral-large": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "ministral-3b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "ministral-8b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "ministral-14b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "pixtral-large": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "pixtral-12b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "open-mistral-nemo": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Mistral AI models are accessed via API only.
local_mode_group = {}


class MistralAILLM(AbstractLLM):
    """LLM implementation for Mistral AI models.

    Provides text summarization using Mistral AI's model family via the
    official Mistral SDK. Supports the full range of Mistral models including
    dense models, mixture-of-experts (Mixtral), and vision models (Pixtral).

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
    """

    def __init__(self, config: MistralAIConfig):
        """Initialize the Mistral AI LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Mistral AI model via the official SDK to generate
        a condensed summary. Sends the text as a user message to the chat
        completion endpoint.

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
                    chat_package = self.client.chat.complete(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    summary = chat_package.choices[0].message.content
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the Mistral AI client for model inference.

        Creates a Mistral client instance using the API key from the
        MISTRALAI_API_KEY environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"MistralAI API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = Mistral(api_key=api_key)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
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
        """Close the Mistral AI client connection.

        Currently a no-op as the Mistral client does not require explicit cleanup.
        """
        pass