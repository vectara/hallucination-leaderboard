"""Moonshot AI (Kimi) model implementations for hallucination evaluation.

This module provides the LLM implementation for Moonshot AI's Kimi model family,
supporting API-based inference via the Moonshot AI platform using an OpenAI-compatible
client. Also supports Together AI as an alternative backend for certain configurations.

Classes:
    MoonshotAIConfig: Configuration model for Moonshot AI model settings.
    MoonshotAISummary: Output model for Moonshot AI summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    MoonshotAILLM: Main LLM class implementing AbstractLLM for Moonshot AI models.

Attributes:
    COMPANY: Provider identifier string ("moonshotai").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).

Note:
    Models using HuggingFace Inference API (e.g., Kimi-K2.5) require
    api_type="huggingface" in their config.
"""

import os
from typing import Literal
from enum import Enum, auto


from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from huggingface_hub import InferenceClient
from together import Together
from openai import OpenAI

COMPANY = "moonshotai"
"""str: Provider identifier used for API key lookup and model registration."""


class MoonshotAIConfig(BasicLLMConfig):
    """Configuration model for Moonshot AI Kimi models.

    Extends BasicLLMConfig with Moonshot AI-specific settings for model selection
    and API configuration. Supports Kimi K2 series models including instruct and
    thinking variants.

    Attributes:
        company: Provider identifier, fixed to "moonshotai".
        model_name: Name of the Kimi model variant to use. Includes K2 Instruct
            and K2 Thinking models.
        date_code: Optional version/date identifier for the model. Used to select
            alternative backends (e.g., "0905" for Together AI).
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
        api_type: Backend API to use. "default" for Moonshot AI's OpenAI-compatible
            API, "huggingface" for HuggingFace Inference API (required for Kimi-K2.5).
    """

    company: Literal["moonshotai"] = "moonshotai"
    model_name: Literal[
        "Kimi-K2-Instruct",
        "kimi-k2-thinking",
        "kimi-k2.5-hf",
        "Kimi-K2.5"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["default", "huggingface"] = "default"

class MoonshotAISummary(BasicSummary):
    """Output model for Moonshot AI summarization results.

    Extends BasicSummary with endpoint and api_type tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
        api_type: The backend API used ("default" for Moonshot, "huggingface" for HF).
    """

    endpoint: Literal["chat", "response"] | None = None
    api_type: Literal["default", "huggingface"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the Moonshot AI or
    Together AI platforms with an OpenAI-compatible client.

    Attributes:
        CHAT_DEFAULT: Standard chat completion via OpenAI-compatible endpoint.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        KIMI_K2_INSTRUCT: Kimi K2 Instruct model with backend selection based on date_code.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    KIMI_K2_INSTRUCT = auto()
    KIMI_K2P5_HF = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Moonshot AI models are accessed via API only.

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
# invoke the API. KIMI_K2_INSTRUCT supports backend selection via date_code.
client_mode_group = {
    "Kimi-K2-Instruct": {
        "chat": ClientMode.KIMI_K2_INSTRUCT
    },
    "kimi-k2-thinking": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "kimi-k2.5": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "Kimi-K2.5": {
        "chat": ClientMode.KIMI_K2P5_HF
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Moonshot AI models are accessed via API only.
local_mode_group = {}

class MoonshotAILLM(AbstractLLM):
    """LLM implementation for Moonshot AI Kimi models.

    Provides text summarization using Moonshot AI's Kimi model family via an
    OpenAI-compatible API. Supports the Moonshot AI platform as the primary
    backend with Together AI as an alternative for certain configurations
    (controlled by date_code).

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        huggingface_name: Full model path in HuggingFace format for Together AI.
    """

    def __init__(self, config: MoonshotAIConfig):
        """Initialize the Moonshot AI LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type
        self.huggingface_name = f"moonshotai/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Kimi model via Moonshot AI or Together AI platforms
        to generate a condensed summary. Routes through the appropriate backend
        based on model configuration and date_code settings.

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
                        model = self.model_fullname,
                        messages = [
                            {"role": "user", "content": prepared_text}
                        ],
                        temperature = self.temperature,
                        max_tokens = self.max_tokens
                    )
                    
                    summary = completion.choices[0].message.content

                case ClientMode.KIMI_K2_INSTRUCT:
                    if self.date_code == "0905":
                        client = Together()
                        response = client.chat.completions.create(
                        model=self.huggingface_name,
                            messages=[
                                {
                                "role": "user",
                                "content": prepared_text
                                }
                            ],
                            max_tokens = self.max_tokens,
                            temperature = self.temperature,
                        )
                        summary = response.choices[0].message.content
                    else:
                        messages = [
                            {"role": "user", "content": [{"type": "text", "text":  prepared_text}]}
                        ]
                        response = self.client.chat.completions.create(
                            model=self.huggingface_name,
                            messages=messages,
                            stream=False,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        summary = response.choices[0].message.content
                case ClientMode.KIMI_K2P5_HF:
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text":  prepared_text}]}
                    ]
                    response = self.client.chat.completions.create(
                        model=self.huggingface_name,
                        messages=messages,
                        stream=False,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = response.choices[0].message.content
                    if "</think>" in summary:
                        summary = summary.split("</think>")[1].strip()

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
        """Initialize the Moonshot AI client for Kimi model inference.

        Creates an OpenAI-compatible client instance configured for the
        Moonshot AI platform using the API key from the MOONSHOTAI_API_KEY
        environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "huggingface":
                    api_key = os.getenv("HF_TOKEN")
                    assert api_key is not None, "HF_TOKEN not found in environment variable HF_TOKEN"
                    self.client = InferenceClient(model=self.huggingface_name, token=api_key)
                elif self.api_type == "default":
                    api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                    assert api_key is not None, f"{COMPANY.upper()} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(
                        api_key = api_key,
                        base_url = "https://api.moonshot.ai/v1",
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
        """Close the Moonshot AI client connection.

        Currently a no-op as the OpenAI-compatible client does not require
        explicit cleanup.
        """
        pass