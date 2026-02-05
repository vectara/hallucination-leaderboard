"""TNG Technology Consulting model implementations for hallucination evaluation.

This module provides the LLM implementation for TNG Technology Consulting's
model offerings, supporting API-based inference via their chat.model.tngtech.com
platform using an OpenAI-compatible client. Currently supports the DeepSeek-based
R1T2-Chimera model.

Classes:
    TngTechConfig: Configuration model for TNG Tech model settings.
    TngTechSummary: Output model for TNG Tech summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    TngTechLLM: Main LLM class implementing AbstractLLM for TNG Tech models.

Attributes:
    COMPANY: Provider identifier string ("tngtech").
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

COMPANY = "tngtech"
"""str: Provider identifier used for API key lookup and model registration."""

class TngTechConfig(BasicLLMConfig):
    """Configuration model for TNG Technology Consulting models.

    Extends BasicLLMConfig with TNG Tech-specific settings for model selection
    and API configuration. Supports the DeepSeek-based R1T2-Chimera model via
    their OpenAI-compatible API platform.

    Attributes:
        company: Provider identifier, fixed to "tngtech".
        model_name: Name of the TNG Tech model variant to use. Currently
            supports DeepSeek-TNG-R1T2-Chimera.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["tngtech"]
    model_name: Literal[
        "DeepSeek-TNG-R1T2-Chimera"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["default"] = "default"

class TngTechSummary(BasicSummary):
    """Output model for TNG Tech summarization results.

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

    Defines how the model should be invoked when using the TNG Tech platform
    with an OpenAI-compatible client.

    Attributes:
        CHAT_DEFAULT: Use chat.completions.create for conversational generation.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as TNG Tech models are accessed via API only.

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
# invoke the TNG Tech API.
client_mode_group = {
    "DeepSeek-TNG-R1T2-Chimera": {
        "chat": ClientMode.CHAT_DEFAULT,
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates TNG Tech models are accessed via API only.
local_mode_group = {}

class TngTechLLM(AbstractLLM):
    """LLM implementation for TNG Technology Consulting models.

    Provides text summarization using TNG Tech's model offerings via their
    OpenAI-compatible API hosted at chat.model.tngtech.com. Currently supports
    the DeepSeek-based R1T2-Chimera model.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        model_fullname: Full model path in provider/model_name format.
    """

    def __init__(self, config: TngTechConfig):
        """Initialize the TNG Tech LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type
        self.model_fullname = f"{COMPANY}/{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured TNG Tech model via their OpenAI-compatible API
        to generate a condensed summary. Sends the text as a user message
        to the chat completions endpoint.

        Args:
            prepared_text: The preprocessed text to summarize.

        Returns:
            The generated summary text, or an error placeholder if generation fails.

        Raises:
            Exception: If neither client nor local_model is initialized, or if
                the model does not support the configured endpoint.
        """
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content":prepared_text}]
                    )
                    summary = chat_package.choices[0].message.content
                case None:
                    raise Exception(f"Model `{self.model_name}` cannot be run from `{self.endpoint}` endpoint")
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the TNG Tech API client for model inference.

        Creates an OpenAI-compatible client instance configured for the
        TNG Tech platform (chat.model.tngtech.com) using the API key from
        the TNGTECH_API_KEY environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "default":
                    api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                    assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://chat.model.tngtech.com/v1/"
                    )
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
        """Close the TNG Tech API client connection.

        Currently a no-op as the OpenAI-compatible client does not require
        explicit cleanup.
        """
        pass