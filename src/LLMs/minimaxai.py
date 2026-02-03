"""MiniMax AI model implementations for hallucination evaluation.

This module provides the LLM implementation for MiniMax AI's model family,
supporting API-based inference via the Fireworks AI platform using an
OpenAI-compatible client.

Classes:
    MiniMaxAIConfig: Configuration model for MiniMax AI model settings.
    MiniMaxAISummary: Output model for MiniMax AI summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    MiniMaxAILLM: Main LLM class implementing AbstractLLM for MiniMax AI models.

Attributes:
    COMPANY: Provider identifier string ("MiniMaxAI").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes.
"""

import os
from typing import Literal
from openai import OpenAI

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
from enum import Enum, auto

COMPANY = "MiniMaxAI"
"""str: Provider identifier used for model registration."""


class MiniMaxAIConfig(BasicLLMConfig):
    """Configuration model for MiniMax AI models.

    Extends BasicLLMConfig with MiniMax AI-specific settings for model selection
    and execution mode configuration.

    Attributes:
        company: Provider identifier, fixed to "MiniMaxAI".
        model_name: Name of the MiniMax AI model variant to use.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference ("api", "cpu", or "gpu").
        endpoint: API endpoint type ("chat" for conversational format).
        api_type: Backend API to use. "default" uses Fireworks AI.
    """

    company: Literal["MiniMaxAI"] = "MiniMaxAI"
    model_name: Literal[
        "minimax-m2p1",
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["default"] = "default"

class MiniMaxAISummary(BasicSummary):
    """Output model for MiniMax AI summarization results.

    Extends BasicSummary with endpoint and api_type tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
        api_type: The backend API used ("default" for Fireworks AI).
    """

    endpoint: Literal["chat", "response"] | None = None
    api_type: Literal["default"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the Fireworks AI
    platform with an OpenAI-compatible client.

    Attributes:
        CHAT_DEFAULT: Standard chat completion endpoint.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
        M2P1: MiniMax M2P1 model via Fireworks AI.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()
    M2P1 = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values.
client_mode_group = {
    "minimax-m2p1": {
        "chat": ClientMode.M2P1
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Contains placeholder entry; local execution not fully implemented.
local_mode_group = {
    "MODEL_NAME": {
        "chat": LocalMode.UNDEFINED
    }
} 

class MiniMaxAILLM(AbstractLLM):
    """LLM implementation for MiniMax AI models.

    Provides text summarization using MiniMax AI's model family via the
    Fireworks AI platform. Uses an OpenAI-compatible client for API calls.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs ("api", "cpu", or "gpu").
        full_config: Complete configuration object for reference.
    """

    def __init__(self, config: MiniMaxAIConfig):
        """Initialize the MiniMax AI LLM with the given configuration.

        Args:
            config: Configuration object specifying model and execution settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured MiniMax AI model via the Fireworks AI platform
        to generate a condensed summary. Routes through the OpenAI-compatible
        chat completions API.

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
                case ClientMode.M2P1:
                    self.model_fullname = f"accounts/fireworks/models/{self.model_name}"
                    response = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prepared_text,
                            }
                        ],
                        model=self.model_fullname,
                    )

                    summary = response.choices[0].message.content
                case ClientMode.CHAT_DEFAULT:
                    pass
        elif self.local_model: 
            match local_mode_group[self.model_name][self.endpoint]:
                case LocalMode.CHAT_DEFAULT:
                    pass
        else:
            raise Exception(
                ModelInstantiationError.MISSING_SETUP.format(
                    class_name=self.__class__.__name__
                )
            )
        return summary

    def setup(self):
        """Initialize the Fireworks AI client for MiniMax model inference.

        Creates an OpenAI-compatible client instance configured for the
        Fireworks AI platform using the API key from the FIREWORKS_API_KEY
        environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                # default api_type uses Fireworks AI
                api_key = os.getenv(f"FIREWORKS_API_KEY")
                assert api_key is not None, f"FIREWORKS API key not found in environment variable FIREWORKS_API_KEY"
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.fireworks.ai/inference/v1"
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
            if self.model_name in local_mode_group:
                self.local_model = None
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )

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
        """Close the Fireworks AI client connection.

        Currently a no-op as the OpenAI-compatible client does not require
        explicit cleanup.
        """
        pass