"""Cohere (Command, Aya) model implementations for hallucination evaluation.

This module provides the LLM implementation for Cohere's Command and Aya model
families, supporting API-based inference via the official Cohere Python SDK.

Classes:
    CohereConfig: Configuration model for Cohere model settings.
    CohereSummary: Output model for Cohere summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes (currently unused).
    CohereLLM: Main LLM class implementing AbstractLLM for Cohere models.

Attributes:
    COMPANY: Provider identifier string ("CohereLabs").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

import cohere

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "CohereLabs"
"""str: Provider identifier used for API key lookup and model registration."""


class CohereConfig(BasicLLMConfig):
    """Configuration model for Cohere Command and Aya models.

    Extends BasicLLMConfig with Cohere-specific settings for model selection
    and API configuration. Supports both the Command series (including reasoning
    variants) and the multilingual Aya Expanse series.

    Attributes:
        company: Provider identifier, fixed to "CohereLabs".
        model_name: Name of the Cohere model variant to use. Includes Command
            models (standard and reasoning) and Aya multilingual models.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["CohereLabs"] = "CohereLabs"
    model_name: Literal[
        "aya-expanse-8b",
        "aya-expanse-32b",
        "c4ai-command-r-plus",
        "command",
        "command-chat",
        "command-a",
        "command-a-reasoning",
        "c4ai-aya-expanse-32b",
        "c4ai-aya-expanse-8b",
        "command-r-plus",
        "command-r",
        "command-r7b"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["default"] = "default"

class CohereSummary(BasicSummary):
    """Output model for Cohere summarization results.

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
    """Execution modes for Cohere API client inference.

    Defines how the model should be invoked when using the Cohere SDK.

    Attributes:
        CHAT_DEFAULT: Standard chat completion, extracts first text content.
        CHAT_REASONING: Chat completion for reasoning models, iterates through
            content blocks to find text response.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    CHAT_REASONING = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Cohere models only support API inference.

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
# invoke the Cohere API. Reasoning models use CHAT_REASONING for proper
# response extraction.
client_mode_group = {
    "command-a": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "command-a-reasoning": {
        "chat": ClientMode.CHAT_REASONING
    },
    "c4ai-aya-expanse-32b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "c4ai-aya-expanse-8b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "command-r-plus": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "command-r": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "command-r7b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Cohere models do not support local execution.
local_mode_group = {}

class CohereLLM(AbstractLLM):
    """LLM implementation for Cohere Command and Aya models.

    Provides text summarization using Cohere's Command and Aya model families
    via the official Cohere Python SDK (V2 client). Supports both standard
    chat completion and reasoning model response extraction.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        full_config: Complete configuration object for reference.
    """

    def __init__(self, config: CohereConfig):
        """Initialize the Cohere LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Cohere model via the Cohere V2 API to generate a
        condensed summary. For reasoning models, iterates through content blocks
        to extract the text response.

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
                    response = self.client.chat(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )

                    summary = response.message.content[0].text
                case ClientMode.CHAT_REASONING:
                    response = self.client.chat(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    for content in response.message.content:
                        if content.type == "text":
                            summary = content.text
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
        """Initialize the Cohere API client for inference.

        Creates a Cohere V2 client instance using the API key from the
        COHERE_API_KEY environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "default":
                    api_key = os.getenv(f"COHERE_API_KEY")
                    assert api_key is not None, (
                        f"{COMPANY} API key not found in environment variable "
                        f"{COMPANY.upper()}_API_KEY"
                    )
                    self.client = cohere.ClientV2(api_key=api_key)
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
        """Close the Cohere API client connection.

        Currently a no-op as the Cohere client does not require explicit cleanup.
        """
        pass