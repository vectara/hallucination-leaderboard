"""Meta (Llama) model implementations for hallucination evaluation.

This module provides the LLM implementation for Meta's Llama model family,
supporting API-based inference via the Together AI platform. Includes support
for Llama 2, 3, 3.1, 3.2, 3.3, and Llama 4 model series.

Classes:
    MetaLlamaConfig: Configuration model for Llama model settings.
    MetaLlamaSummary: Output model for Llama summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes (currently unused).
    MetaLlamaLLM: Main LLM class implementing AbstractLLM for Llama models.

Attributes:
    COMPANY: Provider identifier string ("meta-llama").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from together import Together

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "meta-llama"
"""str: Provider identifier used for model path construction and registration."""


class MetaLlamaConfig(BasicLLMConfig):
    """Configuration model for Meta Llama models.

    Extends BasicLLMConfig with Llama-specific settings for model selection
    and API configuration. Supports the full range of Llama models from
    Llama 2 through Llama 4, including chat, instruct, and vision variants.

    Attributes:
        company: Provider identifier, fixed to "meta-llama".
        model_name: Name of the Llama model variant to use. Includes Llama 2/3/4
            series with various sizes (7B to 405B) and capabilities (chat,
            instruct, vision, turbo).
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" or "response" for completion).
    """

    company: Literal["meta-llama"] = "meta-llama"
    model_name: Literal[
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-70b-chat-hf",
        "Llama-3-8B-chat-hf",
        "Llama-3-70B-chat-hf",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct-Turbo",
        "Llama-3.2-11B-Vision-Instruct-Turbo",
        "Llama-3.2-90B-Vision-Instruct-Turbo",
        "Llama-3.3-70B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",

        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Llama-4-Scout-17B-16E-Instruct",
        "Meta-Llama-3.1-8B-Instruct-Turbo",
        "Llama-3.3-70B-Instruct-Turbo",
        "Llama-3.3-70B-Instruct-Turbo-Free",
        "Meta-Llama-3.1-405B-Instruct-Turbo",
        "Llama-3.2-3B-Instruct-Turbo",
        "Llama-3.2-11B-Vision-Instruct-Turbo*",
        "Llama-3.2-90B-Vision-Instruct-Turbo*",
        "Meta-Llama-3.1-405B-Instruct-Turbo",
        "Meta-Llama-3.1-8B-Instruct-Turbo",
        "Meta-Llama-3-8B-Instruct-Lite",
        "Llama-3-8b-chat-hf*",
        "Llama-3-70b-chat-hf",
        "Llama-2-70b-hf"  # Completion?
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class MetaLlamaSummary(BasicSummary):
    """Output model for Meta Llama summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for Together AI API client inference.

    Defines how the model should be invoked when using the Together AI SDK.

    Attributes:
        CHAT_DEFAULT: Use chat.completions.create for conversational models.
        RESPONSE_DEFAULT: Use completions.create for base/completion models.
        UNDEFINED: Mode not defined or not supported (e.g., vision models).
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Llama models are accessed via Together AI API.

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
# invoke the Together AI API. Chat models use CHAT_DEFAULT, base models use
# RESPONSE_DEFAULT. Vision models marked with * are currently UNDEFINED.
client_mode_group = {
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Llama-4-Scout-17B-16E-Instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Meta-Llama-3.1-8B-Instruct-Turbo": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Llama-3.3-70B-Instruct-Turbo": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Llama-3.3-70B-Instruct-Turbo-Free": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Meta-Llama-3.1-405B-Instruct-Turbo": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Llama-3.2-3B-Instruct-Turbo": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Llama-3.2-11B-Vision-Instruct-Turbo*": {  # Unable to access model atm
        "chat": ClientMode.UNDEFINED
    },
    "Llama-3.2-90B-Vision-Instruct-Turbo*": {  # Unable to access model atm
        "chat": ClientMode.UNDEFINED
    },
    "Meta-Llama-3.1-405B-Instruct-Turbo": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Meta-Llama-3.1-8B-Instruct-Turbo": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Meta-Llama-3-8B-Instruct-Lite": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Llama-3-8b-chat-hf*": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Llama-3-70b-chat-hf": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Llama-2-70b-hf": {
        "response": ClientMode.RESPONSE_DEFAULT
    }  # Completion?
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Llama models are accessed via Together AI API only.
local_mode_group = {}

class MetaLlamaLLM(AbstractLLM):
    """LLM implementation for Meta Llama models.

    Provides text summarization using Meta's Llama model family via the
    Together AI API platform. Supports both chat completion and text
    completion modes depending on the model variant.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat" or "response").
        execution_mode: Where inference runs (currently only "api" supported).
    """

    def __init__(self, config: MetaLlamaConfig):
        """Initialize the Meta Llama LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Llama model via the Together AI API to generate
        a condensed summary. Routes to chat completion or text completion
        based on the model's configured mode.

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
                    together_name = f"meta-llama/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                        model=together_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    summary = response.choices[0].message.content

                case ClientMode.RESPONSE_DEFAULT:
                    together_name = f"meta-llama/{self.model_fullname}"
                    response = self.client.completions.create(
                        model=together_name,
                        prompt=prepared_text,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    summary = response.choices[0].text
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
        """Initialize the Together AI client for Llama model inference.

        Creates a Together client instance using the API key from the
        TOGETHER_API_KEY environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"TOGETHER_API_KEY")
                assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = Together(api_key=api_key)
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
        """Close the Together AI client connection.

        Currently a no-op as the Together client does not require explicit cleanup.
        """
        pass