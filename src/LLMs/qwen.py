"""Qwen (Alibaba Cloud) model implementations for hallucination evaluation.

This module provides the LLM implementation for Alibaba Cloud's Qwen model family,
supporting API-based inference via the DashScope platform using an OpenAI-compatible
client. Includes support for Qwen2, Qwen2.5, Qwen3, and QwQ reasoning models with
optional thinking/reasoning mode.

Classes:
    QwenConfig: Configuration model for Qwen model settings.
    QwenSummary: Output model for Qwen summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    QwenLLM: Main LLM class implementing AbstractLLM for Qwen models.

Attributes:
    COMPANY: Provider identifier string ("qwen").
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

COMPANY = "qwen"
"""str: Provider identifier used for API key lookup and model registration."""


class QwenConfig(BasicLLMConfig):
    """Configuration model for Qwen models.

    Extends BasicLLMConfig with Qwen-specific settings for model selection,
    API configuration, and thinking/reasoning mode. Supports the full range
    of Qwen models from Qwen2 through Qwen3, including vision and reasoning
    variants.

    Attributes:
        company: Provider identifier, fixed to "qwen".
        model_name: Name of the Qwen model variant to use. Includes Qwen2/2.5/3
            series with various sizes (0.5B to 235B), vision models (VL), and
            reasoning models (QwQ, thinking variants).
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
        thinking_tokens: Whether to include thinking tokens in output.
        enable_thinking: Whether to enable thinking/reasoning mode for supported
            models. Passed via extra_body to the API.
    """

    company: Literal["qwen"]
    model_name: Literal[
        "Qwen2-72B-Instruct",
        "Qwen2-VL-2B-Instruct",
        "Qwen2-VL-7B-Instruct",
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-30B-A3B",
        "Qwen3-32B",
        "Qwen3-235B-A22B",
        "QwQ-32B-Preview",

        "qwen3-30b-a3b-thinking",
        "qwen3-next-80b-a3b-thinking",
        "qwen3-omni-30b-a3b-thinking",

        "qwen3-max-preview",
        "qwen3-32b",
        "qwen3-14b",
        "qwen3-8b",
        "qwen3-4b",
        "qwen3-1.7b",
        "qwen3-0.6b",
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "qwen2.5-72b-instruct",
        "qwen2.5-32b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-7b-instruct",
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    thinking_tokens: bool = None
    enable_thinking: bool = None

class QwenSummary(BasicSummary):
    """Output model for Qwen summarization results.

    Extends BasicSummary with endpoint and thinking mode tracking for
    result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
        enable_thinking: Whether thinking mode was enabled during generation.
    """

    endpoint: Literal["chat", "response"] | None = None
    enable_thinking: bool | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the DashScope platform
    with an OpenAI-compatible client.

    Attributes:
        CHAT_DEFAULT: Standard chat completion without thinking mode.
        CHAT_REASONING: Chat completion with enable_thinking support via extra_body.
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
    Currently unused as Qwen models are accessed via DashScope API only.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values. Models with CHAT_REASONING
# support the enable_thinking parameter for reasoning/thinking mode.
client_mode_group = {
    "Qwen3-235B-A22B": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-30b-a3b-thinking": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-next-80b-a3b-thinking": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-omni-30b-a3b-thinking": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-max-preview": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-32b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-14b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-8b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-4b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-1.7b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-0.6b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen-plus": {
        "chat": ClientMode.CHAT_REASONING
    }, # 2025-04-28
    "qwen-turbo": {
        "chat": ClientMode.CHAT_REASONING
    }, # 2025-04-28
    "Qwen2.5-Max": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen-max": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen2.5-72b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen2.5-32b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen2.5-14b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen2.5-7b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Qwen models are accessed via DashScope API only.
local_mode_group = {}


class QwenLLM(AbstractLLM):
    """LLM implementation for Qwen models.

    Provides text summarization using Alibaba Cloud's Qwen model family via
    the DashScope platform's OpenAI-compatible API. Supports optional
    thinking/reasoning mode for compatible models.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        enable_thinking: Whether thinking mode is enabled for reasoning models.
    """

    def __init__(self, config: QwenConfig):
        """Initialize the Qwen LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.enable_thinking = config.enable_thinking

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Qwen model via the DashScope API to generate
        a condensed summary. For reasoning models, passes enable_thinking
        via extra_body to control thinking/reasoning mode.

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
                    completion = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "user", "content": prepared_text}],
                        )
                    summary = completion.choices[0].message.content
                case ClientMode.CHAT_REASONING:
                    completion = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_body = {"enable_thinking": self.enable_thinking},
                        messages=[
                            {"role": "user", "content": prepared_text}],
                        )
                    summary = completion.choices[0].message.content
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
        """Initialize the DashScope API client for Qwen model inference.

        Creates an OpenAI-compatible client instance configured for the
        DashScope platform using the API key from the QWEN_API_KEY
        environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = OpenAI(
                    api_key=api_key, 
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
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
        """Close the DashScope API client connection.

        Currently a no-op as the OpenAI-compatible client does not require
        explicit cleanup.
        """
        pass