"""xAI (Grok) model implementations for hallucination evaluation.

This module provides the LLM implementation for xAI's Grok model family,
supporting API-based inference via the official xai_sdk. Includes support
for Grok-2, Grok-3, and Grok-4 series models with optional reasoning mode
for Grok-4 variants.

Classes:
    XAIConfig: Configuration model for xAI model settings.
    XAISummary: Output model for xAI summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    XAILLM: Main LLM class implementing AbstractLLM for xAI models.

Attributes:
    COMPANY: Provider identifier string ("xai-org").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from xai_sdk import Client
from xai_sdk.chat import user, system
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "xai-org"
"""str: Provider identifier used for API key lookup and model registration."""


class XAIConfig(BasicLLMConfig):
    """Configuration model for xAI Grok models.

    Extends BasicLLMConfig with xAI-specific settings for model selection,
    API configuration, and reasoning mode. Supports the full Grok model family
    from Grok-2 through Grok-4, including vision and reasoning variants.

    Attributes:
        company: Provider identifier, fixed to "xai-org".
        model_name: Name of the Grok model variant to use. Includes Grok-2
            vision, Grok-3 series (standard, mini, fast), and Grok-4 series
            with reasoning/non-reasoning variants.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
        reasoning_effort: Reasoning intensity for Grok-4 models ("NA", "low",
            or "high"). Set to "NA" for non-reasoning models.
    """

    company: Literal["xai-org"] = "xai-org"
    model_name: Literal[
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
        "grok-4",
        "grok-3",
        "grok-3-mini",
        "grok-3-fast",
        "grok-3-mini-fast",
        "grok-2-vision"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    reasoning_effort: Literal["NA", "low", "high"] = "NA"

class XAISummary(BasicSummary):
    """Output model for xAI summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the xAI SDK.

    Attributes:
        CHAT_DEFAULT: Standard chat completion for Grok-2/3 models.
        CHAT_REASONING: Chat completion with reasoning token tracking for Grok-4.
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
    Currently unused as xAI models are accessed via API only.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values. Grok-4 models use
# CHAT_REASONING for reasoning token tracking, while Grok-2/3 use CHAT_DEFAULT.
client_mode_group = {
    "grok-4-1-fast-reasoning": {
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4-1-fast-non-reasoning": {
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4-fast-non-reasoning": {
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4-fast-reasoning": {
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4-fast-non-reasoning": {
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4": {
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-3": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "grok-3-mini": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "grok-3-fast": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "grok-3-mini-fast": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "grok-2-vision": {
        "chat": ClientMode.CHAT_DEFAULT
    },
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates xAI models are accessed via API only.
local_mode_group = {}

class XAILLM(AbstractLLM):
    """LLM implementation for xAI Grok models.

    Provides text summarization using xAI's Grok model family via the official
    xai_sdk. Supports standard chat completion for Grok-2/3 models and reasoning
    mode with token tracking for Grok-4 variants.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        reasoning_effort: Reasoning intensity for Grok-4 models.
        full_config: Complete configuration object for reference.
    """

    def __init__(self, config: XAIConfig):
        """Initialize the xAI LLM with the given configuration.

        Args:
            config: Configuration object specifying model, API settings,
                and reasoning mode configuration.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.reasoning_effort = config.reasoning_effort
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Grok model via the xAI SDK to generate a condensed
        summary. For Grok-4 models with CHAT_REASONING mode, tracks reasoning
        tokens in self.thinking_tokens. Uses the SDK's chat interface with
        user message formatting.

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
                case ClientMode.CHAT_REASONING:
                    chat = self.client.chat.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    chat.append(user(prepared_text))

                    response = chat.sample()
                    summary = response.content
                    self.thinking_tokens = response.usage.reasoning_tokens
                case ClientMode.CHAT_DEFAULT:
                    chat = self.client.chat.create(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    chat.append(user(prepared_text))

                    response = chat.sample()
                    summary = response.content
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
        """Initialize the xAI SDK client for model inference.

        Creates an xAI Client instance configured for the api.x.ai endpoint
        using the API key from the XAI_API_KEY environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"XAI_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = Client(
                    api_host="api.x.ai",
                    api_key=api_key
                )
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
        """Close the xAI SDK client connection.

        Currently a no-op as the xAI Client does not require
        explicit cleanup.
        """
        pass