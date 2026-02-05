"""DeepSeek AI model implementations for hallucination evaluation.

This module provides the LLM implementation for DeepSeek AI's model family,
supporting API-based inference via the HuggingFace Inference API. Includes
support for the DeepSeek-V2, V3, and R1 (reasoning) model series.

Classes:
    DeepSeekAIConfig: Configuration model for DeepSeek model settings.
    DeepSeekAISummary: Output model for DeepSeek summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes (currently unused).
    DeepSeekAILLM: Main LLM class implementing AbstractLLM for DeepSeek models.

Attributes:
    COMPANY: Provider identifier string ("deepseek-ai").
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

COMPANY = "deepseek-ai"
"""str: Provider identifier used for model path construction and registration."""

class DeepSeekAIConfig(BasicLLMConfig):
    """Configuration model for DeepSeek AI models.

    Extends BasicLLMConfig with DeepSeek-specific settings for model selection
    and API configuration. Supports the V2, V3, and R1 model series including
    chat and coder variants.

    Attributes:
        company: Provider identifier, fixed to "deepseek-ai".
        model_name: Name of the DeepSeek model variant to use. Includes V3.x
            series, R1 reasoning models, and legacy V2.5.
        execution_mode: Where to run inference, currently only "api" supported.
        date_code: Optional version/date identifier for the model.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["deepseek-ai"] = "deepseek-ai"
    model_name: Literal[
        "DeepSeek-V3.1-Terminus",
        "DeepSeek-V3.2",
        "DeepSeek-V3.2-Exp",
        "deepseek-chat",
        "deepseek-coder",
        "DeepSeek-R1-0528",
        "DeepSeek-V3",
        "DeepSeek-V3.1",
        "DeepSeek-R1",
        "DeepSeek-V2.5"  # Not implemented


    ]
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["huggingface"] = "huggingface"

class DeepSeekAISummary(BasicSummary):
    """Output model for DeepSeek AI summarization results.

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
    """Execution modes for HuggingFace Inference API client.

    Defines how the model should be invoked when using the HuggingFace
    Inference API. Different modes support different parameter combinations.

    Attributes:
        CHAT_DEFAULT: Chat completion with temperature and max_tokens parameters.
        CHAT_NO_TEMP_NO_TOKENS: Chat completion without generation parameters.
        CHAT_CONVERSATIONAL_NO_TOKENS: Conversational API with temperature only.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    CHAT_NO_TEMP_NO_TOKENS = auto()
    CHAT_CONVERSATIONAL_NO_TOKENS = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as DeepSeek models only support API inference.

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
# invoke the HuggingFace Inference API. V2.5 uses a mode without temperature/tokens.
client_mode_group = {
    "DeepSeek-V3.2": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-R1": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V3": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V3.1": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V3.1-Terminus": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V3.2-Exp": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V2.5": {
        "chat": ClientMode.CHAT_NO_TEMP_NO_TOKENS
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates DeepSeek models do not support local execution.
local_mode_group = {}


class DeepSeekAILLM(AbstractLLM):
    """LLM implementation for DeepSeek AI models.

    Provides text summarization using DeepSeek AI's model family via the
    HuggingFace Inference API. Supports multiple generation modes including
    chat completion and conversational APIs.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        model_fullname: Full HuggingFace model path (e.g., "deepseek-ai/DeepSeek-V3").
    """

    def __init__(self, config: DeepSeekAIConfig):
        """Initialize the DeepSeek AI LLM with the given configuration.

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

        Uses the configured DeepSeek model via the HuggingFace Inference API
        to generate a condensed summary. Supports multiple API modes:
        - CHAT_DEFAULT: Full chat completion with temperature and max_tokens
        - CHAT_NO_TEMP_NO_TOKENS: Chat completion without generation parameters
        - CHAT_CONVERSATIONAL_NO_TOKENS: Conversational API with temperature only

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
                    messages = [{"role": "user", "content": prepared_text}]
                    client_package = self.client.chat_completion(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = client_package.choices[0].message.content
                case ClientMode.CHAT_NO_TEMP_NO_TOKENS:
                    messages = [{"role": "user", "content": prepared_text}]
                    client_package = self.client.chat_completion(
                        messages
                    )
                    summary = client_package.choices[0].message.content
                case ClientMode.CHAT_CONVERSATIONAL_NO_TOKENS:
                    client_package = self.client.conversational(
                        messages=prepared_text,
                        temperature=self.temperature
                    )
                    summary = client_package["generated_text"]
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the HuggingFace Inference client for DeepSeek inference.

        Creates an InferenceClient instance configured for the specified
        DeepSeek model using the HF_TOKEN environment variable for
        authentication with the HuggingFace Inference API.

        Raises:
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "huggingface":
                    api_key = os.getenv("HF_TOKEN")
                    assert api_key is not None, "HF_TOKEN not found in environment variable HF_TOKEN"
                    self.client = InferenceClient(model=self.model_fullname, token=api_key)
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
