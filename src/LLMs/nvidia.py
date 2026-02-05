"""NVIDIA (Nemotron) model implementations for hallucination evaluation.

This module provides the LLM implementation for NVIDIA's Nemotron model family,
supporting API-based inference via DeepInfra (primary) and Replicate (alternative)
platforms using an OpenAI-compatible client.

Classes:
    NvidiaConfig: Configuration model for NVIDIA model settings.
    NvidiaSummary: Output model for NVIDIA summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    NvidiaLLM: Main LLM class implementing AbstractLLM for NVIDIA models.

Attributes:
    COMPANY: Provider identifier string ("nvidia").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes.
"""

import os
from typing import Literal
import re
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

import replicate
from openai import OpenAI

COMPANY = "nvidia"
"""str: Provider identifier used for model path construction and registration."""


class NvidiaConfig(BasicLLMConfig):
    """Configuration model for NVIDIA Nemotron models.

    Extends BasicLLMConfig with NVIDIA-specific settings for model selection
    and execution mode configuration. Supports the Nemotron model family
    via DeepInfra or Replicate APIs.

    Attributes:
        company: Provider identifier, fixed to "nvidia".
        model_name: Name of the NVIDIA model variant to use. Currently supports
            Nemotron-3-Nano-30B-A3B.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference ("api", "cpu", or "gpu").
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["nvidia"] = "nvidia"
    model_name: Literal[
        "Nemotron-3-Nano-30B-A3B",
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["deepinfra"] = "deepinfra"

class NvidiaSummary(BasicSummary):
    """Output model for NVIDIA summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None
    api_type: Literal["deepinfra"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using DeepInfra or Replicate
    platforms for NVIDIA model inference.

    Attributes:
        CHAT_DEFAULT: Standard chat completion via OpenAI-compatible endpoint.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        REPLICATE_NEMOTRON_3_NANO_30B_A3B: Nemotron via Replicate with thinking support.
        DEEPINFRA_NEMOTRON_3_NANO_30B_A3B: Nemotron via DeepInfra OpenAI-compatible API.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    REPLICATE_NEMOTRON_3_NANO_30B_A3B = auto()
    DEEPINFRA_NEMOTRON_3_NANO_30B_A3B = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently a placeholder as local execution is not fully implemented.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values indicating which
# backend platform to use (DeepInfra or Replicate).
client_mode_group = {
    "Nemotron-3-Nano-30B-A3B": {
        "chat": ClientMode.DEEPINFRA_NEMOTRON_3_NANO_30B_A3B
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Contains placeholder entry; local execution not fully implemented.
local_mode_group = {
    "MODEL_NAME": {
        "chat": LocalMode.UNDEFINED
    }
} 

class NvidiaLLM(AbstractLLM):
    """LLM implementation for NVIDIA Nemotron models.

    Provides text summarization using NVIDIA's Nemotron model family via
    DeepInfra (primary) or Replicate (alternative) platforms. Supports
    thinking mode with automatic stripping of thinking tags from output.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs ("api", "cpu", or "gpu").
        full_config: Complete configuration object for reference.
    """

    def __init__(self, config: NvidiaConfig):
        """Initialize the NVIDIA LLM with the given configuration.

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

        Uses the configured Nemotron model via DeepInfra or Replicate to generate
        a condensed summary. For Replicate backend, supports thinking mode and
        automatically strips thinking tags from the output.

        Args:
            prepared_text: The preprocessed text to summarize.

        Returns:
            The generated summary text with thinking tags removed, or an error
            placeholder if generation fails.

        Raises:
            Exception: If neither client nor local_model is initialized.
        """

        def strip_thinking(text: str) -> str:
            text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
            return text.strip()

        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.REPLICATE_NEMOTRON_3_NANO_30B_A3B:
                    replicate_name = f"{COMPANY}/{self.model_fullname}:135b4a9c545002830563436c88ea56b401d135faa59da6773bc5934d2ae56344"
                    output = replicate.run(
                        replicate_name,
                        input={
                            "prompt": prepared_text,
                            "temperature": self.temperature,
                            "max_new_tokens": self.max_tokens,
                            "enable_thinking": True,
                        }
                    )
                    chunks = []

                    for item in output:
                        chunks.append(item if isinstance(item, str) else item.get("text", ""))

                    raw_text = "".join(chunks)
                    summary = strip_thinking(raw_text)
                case ClientMode.DEEPINFRA_NEMOTRON_3_NANO_30B_A3B:
                    deep_infra_name = f"{COMPANY}/{self.model_fullname}"
                    chat_completion = self.client.chat.completions.create(
                        model=deep_infra_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = chat_completion.choices[0].message.content.lstrip()
        elif self.local_model: 
            match local_mode_group[self.model_name][self.endpoint]:
                case 1:
                    pass
        else:
            raise Exception(
                ModelInstantiationError.MISSING_SETUP.format(
                    class_name=self.__class__.__name__
                )
            )
        return summary

    def setup(self):
        """Initialize the API client for NVIDIA model inference.

        Creates an OpenAI-compatible client instance configured for the
        DeepInfra platform using the API key from the DEEPINFRA_API_KEY
        environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "deepinfra":
                    api_key = os.getenv(f"DEEPINFRA_API_KEY")
                    assert api_key is not None, (
                        f"{COMPANY} API key not found in environment variable "
                    )

                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepinfra.com/v1/openai",
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
            if self.model_name in local_mode_group:
                # TODO: Assign a local model if using a local model
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
        """Close the API client connection.

        Currently a no-op as the OpenAI-compatible client does not require
        explicit cleanup.
        """
        pass