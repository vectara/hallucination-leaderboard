"""Arcee AI model implementations for hallucination evaluation.

This module provides the LLM implementation for Arcee AI's Trinity model family,
supporting API-based inference via the Arcee AI platform using an OpenAI-compatible
client.

Classes:
    ArceeAIConfig: Configuration model for Arcee AI model settings.
    ArceeAISummary: Output model for Arcee AI summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    ArceeAILLM: Main LLM class implementing AbstractLLM for Arcee AI models.

Attributes:
    COMPANY: Provider identifier string ("arcee-ai").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from openai import OpenAI

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "arcee-ai"
"""str: Provider identifier used for API key lookup and model registration."""


class ArceeAIConfig(BasicLLMConfig):
    """Configuration model for Arcee AI Trinity models.

    Extends BasicLLMConfig with Arcee AI-specific settings for model selection
    and API configuration.

    Attributes:
        company: Provider identifier, fixed to "arcee-ai".
        model_name: Name of the Trinity model variant to use.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["arcee-ai"] = "arcee-ai"
    model_name: Literal[
        "trinity-large-preview",
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["default"] = "default"


class ArceeAISummary(BasicSummary):
    """Output model for Arcee AI summarization results.

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
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the Arcee AI platform
    with an OpenAI-compatible client.

    Attributes:
        CHAT_DEFAULT: Standard chat completion via OpenAI-compatible endpoint.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Arcee AI models are accessed via API only.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


client_mode_group = {
    "trinity-large-preview": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}
"""dict: Mapping of model names to their supported API client modes."""

local_mode_group = {}
"""dict: Mapping of model names to local execution modes. Empty as Arcee AI models are API-only."""


class ArceeAILLM(AbstractLLM):
    """LLM implementation for Arcee AI Trinity models.

    Provides text summarization using Arcee AI's Trinity model family via an
    OpenAI-compatible API.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
    """

    def __init__(self, config: ArceeAIConfig):
        """Initialize the Arcee AI LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Trinity model via Arcee AI platform to generate
        a condensed summary.

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
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prepared_text}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False
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
        """Initialize the Arcee AI client for Trinity model inference.

        Creates an OpenAI-compatible client instance configured for the
        Arcee AI platform using the API key from the ARCEE_AI_API_KEY
        environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "default":
                    api_key = os.getenv("ARCEE_AI_API_KEY")
                    assert api_key is not None, (
                        "Arcee AI API key not found in environment variable ARCEE_AI_API_KEY"
                    )
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.arcee.ai/api/v1"
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
        """Close the Arcee AI client connection.

        Currently a no-op as the OpenAI-compatible client does not require
        explicit cleanup.
        """
        pass
