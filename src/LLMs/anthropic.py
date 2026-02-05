"""Anthropic (Claude) model implementations for hallucination evaluation.

This module provides the LLM implementation for Anthropic's Claude model family,
supporting API-based inference via the official Anthropic Python SDK.

Classes:
    AnthropicConfig: Configuration model for Claude model settings.
    AnthropicSummary: Output model for Claude summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes (currently unused).
    AnthropicLLM: Main LLM class implementing AbstractLLM for Claude models.

Attributes:
    COMPANY: Provider identifier string ("anthropic").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

import anthropic

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "anthropic"
"""str: Provider identifier used for API key lookup and model registration."""


class AnthropicConfig(BasicLLMConfig):
    """Configuration model for Anthropic Claude models.

    Extends BasicLLMConfig with Anthropic-specific settings for model selection
    and API configuration. Supports the full range of Claude model variants
    from Claude 2 through Claude 4.

    Attributes:
        company: Provider identifier, fixed to "anthropic".
        model_name: Name of the Claude model variant to use. Includes opus,
            sonnet, and haiku tiers across multiple generations.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for messages API).
    """

    company: Literal["anthropic"] = "anthropic"
    model_name: Literal[
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-1",
        "claude-haiku-4-5",
        "claude-3-5-haiku",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "claude-3-sonnet",
        "claude-3-opus",
        "claude-2.0"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["default"] = "default"

    class Config:
        """Pydantic configuration to forbid extra fields during parsing."""

        extra = "forbid"

class AnthropicSummary(BasicSummary):
    """Output model for Anthropic Claude summarization results.

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
    """Execution modes for Anthropic API client inference.

    Defines how the model should be invoked when using the Anthropic SDK.

    Attributes:
        CHAT_DEFAULT: Use the messages.create API for chat completions.
        RESPONSE_DEFAULT: Use the completion API endpoint (legacy).
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Claude models only support API inference.

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
# invoke the Anthropic API. All Claude models use the messages API.
client_mode_group = {
    "claude-opus-4-6": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-opus-4-5": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-3-5-haiku": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-sonnet-4-5": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-haiku-4-5": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-opus-4-1": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-opus-4": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-sonnet-4": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-3-7-sonnet": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-3-5-sonnet": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-3-sonnet": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-3-opus": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "claude-2.0": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Claude models do not support local execution.
local_mode_group = {}

class AnthropicLLM(AbstractLLM):
    """LLM implementation for Anthropic Claude models.

    Provides text summarization using Anthropic's Claude model family via the
    official Anthropic Python SDK. Supports all Claude model tiers (opus,
    sonnet, haiku) across multiple generations.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
    """

    def __init__(self, config: AnthropicConfig):
        """Initialize the Anthropic LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Claude model via the Anthropic messages API to
        generate a condensed summary.

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
                    # Use streaming to handle long-running requests (>10 min)
                    with self.client.messages.stream(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    ) as stream:
                        response = stream.get_final_message()
                        summary = response.content[0].text
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the Anthropic API client for inference.

        Creates an Anthropic client instance using the API key from the
        ANTHROPIC_API_KEY environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "default":
                    api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                    assert api_key is not None, f"Anthropic API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = anthropic.Anthropic(api_key=api_key)
                else:
                    raise ValueError(f"Unknown api_type: {self.api_type}")
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
        """Close the Anthropic API client connection.

        Currently a no-op as the Anthropic client does not require explicit cleanup.
        """
        pass