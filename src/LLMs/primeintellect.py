"""Prime Intellect model implementations for hallucination evaluation.

This module provides the LLM implementation for Prime Intellect's model family,
supporting API-based inference via the HuggingFace Inference API. Currently
supports the INTELLECT-3 model.

Classes:
    PrimeIntellectConfig: Configuration model for Prime Intellect model settings.
    PrimeIntellectSummary: Output model for Prime Intellect summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    PrimeIntellectLLM: Main LLM class implementing AbstractLLM for Prime Intellect models.

Attributes:
    COMPANY: Provider identifier string ("PrimeIntellect").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from huggingface_hub import InferenceClient

COMPANY = "PrimeIntellect"
"""str: Provider identifier used for model path construction and registration."""


class PrimeIntellectConfig(BasicLLMConfig):
    """Configuration model for Prime Intellect models.

    Extends BasicLLMConfig with Prime Intellect-specific settings for model
    selection and API configuration. Currently supports the INTELLECT-3 model
    via HuggingFace Inference API.

    Attributes:
        company: Provider identifier, fixed to "PrimeIntellect".
        model_name: Name of the Prime Intellect model variant to use.
            Currently supports INTELLECT-3.
        execution_mode: Where to run inference, currently only "api" supported.
        date_code: Optional version/date identifier for the model.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["PrimeIntellect"] = "PrimeIntellect"
    model_name: Literal[
        "INTELLECT-3"
    ]
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["huggingface"] = "huggingface"

class PrimeIntellectSummary(BasicSummary):
    """Output model for Prime Intellect summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None
    api_type: Literal["huggingface"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the HuggingFace
    Inference API.

    Attributes:
        CHAT_DEFAULT: Use chat_completion for conversational generation.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Prime Intellect models are accessed via API only.

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
# invoke the HuggingFace Inference API.
client_mode_group = {
    "INTELLECT-3": {
        "chat": ClientMode.CHAT_DEFAULT
    },
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Prime Intellect models are accessed via API only.
local_mode_group = {}

class PrimeIntellectLLM(AbstractLLM):
    """LLM implementation for Prime Intellect models.

    Provides text summarization using Prime Intellect's model family via the
    HuggingFace Inference API. Uses the InferenceClient for chat completion
    requests.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        model_fullname: Full model path in HuggingFace format (company/model_name).
    """

    def __init__(self, config: PrimeIntellectConfig):
        """Initialize the Prime Intellect LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type
        self.model_fullname = f"{self.company}/{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Prime Intellect model via the HuggingFace Inference
        API to generate a condensed summary. Sends the text as a user message
        to the chat_completion endpoint.

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
                    messages = [{"role": "user", "content":prepared_text}]
                    client_package = self.client.chat_completion(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = client_package.choices[0].message.content
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the HuggingFace Inference client for Prime Intellect model inference.

        Creates an InferenceClient instance configured for the specified model.
        Uses cached HuggingFace login credentials.

        Raises:
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "huggingface":
                    self.client = InferenceClient(model=self.model_fullname)
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
        """Close the HuggingFace Inference client connection.

        Currently a no-op as the InferenceClient does not require explicit cleanup.
        """
        pass
