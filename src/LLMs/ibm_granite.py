"""IBM Granite model implementations for hallucination evaluation.

This module provides the LLM implementation for IBM's Granite model family,
supporting API-based inference via the Replicate platform and local inference
via HuggingFace transformers. Includes Granite 3.x and 4.x model series.

Classes:
    IBMGraniteConfig: Configuration model for Granite model settings.
    IBMGraniteSummary: Output model for Granite summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    IBMGraniteLLM: Main LLM class implementing AbstractLLM for Granite models.

Attributes:
    COMPANY: Provider identifier string ("ibm-granite").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to supported local execution modes.
"""

import os
import torch
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
import re
import gc
import replicate

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPANY = "ibm-granite"
"""str: Provider identifier used for model path construction and registration."""

class IBMGraniteConfig(BasicLLMConfig):
    """Configuration model for IBM Granite models.

    Extends BasicLLMConfig with Granite-specific settings for model selection
    and execution mode configuration. Supports Granite 3.x instruction-tuned
    models and Granite 4.0 models in various sizes.

    Attributes:
        company: Provider identifier, fixed to "ibm-granite".
        model_name: Name of the Granite model variant to use. Includes 4.0
            hybrid models and 3.x instruction-tuned models (2B and 8B sizes).
        endpoint: API endpoint type ("chat" for conversational format).
        execution_mode: Where to run inference ("api", "gpu", or "cpu").
    """

    company: Literal["ibm-granite"] = "ibm-granite"
    model_name: Literal[
        "granite-4.0-h-small",
        "granite-4.0-h-tiny",
        "granite-4.0-h-micro",
        "granite-4.0-micro",
        "granite-3.3-8b-instruct",
        "granite-3.2-8b-instruct",
        "granite-3.2-2b-instruct",
        "granite-3.1-8b-instruct",
        "granite-3.1-2b-instruct",
        "granite-3.0-8b-instruct",
        "granite-3.0-2b-instruct"
    ]
    endpoint: Literal["chat", "response"] = "chat"
    execution_mode: Literal["api", "gpu", "cpu"] = "api"

class IBMGraniteSummary(BasicSummary):
    """Output model for IBM Granite summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for Replicate API client inference.

    Defines how the model should be invoked when using the Replicate API.

    Attributes:
        CHAT_DEFAULT: Standard chat completion via Replicate.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally via
    HuggingFace transformers.

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
# invoke the Replicate API.
client_mode_group = {
    "granite-4.0-h-small": {
        "chat": ClientMode.CHAT_DEFAULT,
    },
    "granite-3.3-8b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "granite-3.2-8b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },

}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Models are listed but currently use UNDEFINED mode, indicating local inference
# support is planned but not yet fully implemented.
local_mode_group = {
    "granite-4.0-h-small": {
        "chat": ClientMode.UNDEFINED,
    },
    "granite-4.0-h-tiny": {
        "chat": ClientMode.UNDEFINED,
    },
    "granite-4.0-h-micro": {
        "chat": ClientMode.UNDEFINED,
    },
    "granite-4.0-micro": {
        "chat": ClientMode.UNDEFINED,
    },
    "granite-3.2-8b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.2-2b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.1-8b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.1-2b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.0-8b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.0-2b-instruct": {
        "chat": ClientMode.UNDEFINED
    }
}

class IBMGraniteLLM(AbstractLLM):
    """LLM implementation for IBM Granite models.

    Provides text summarization using IBM's Granite model family via the
    Replicate API platform. Supports Granite 3.x instruction-tuned models
    and Granite 4.0 hybrid models. Local inference support is planned but
    not yet fully implemented.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs ("api", "gpu", or "cpu").
        device: PyTorch device for local inference.
        model_fullname: Full Replicate model path (e.g., "ibm-granite/granite-3.3-8b-instruct").
    """

    def __init__(self, config: IBMGraniteConfig):
        """Initialize the IBM Granite LLM with the given configuration.

        Args:
            config: Configuration object specifying model and execution settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_fullname = f"{COMPANY}/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Granite model via the Replicate API to generate
        a condensed summary. Local inference is supported in the interface
        but not yet implemented.

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
                case ClientMode.CHAT_DEFAULT:  # Default
                    input = {
                        "prompt": prepared_text,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                    }
                    raw_out = replicate.run(
                        f"{self.model_fullname}",
                        input=input
                    )
                    summary = raw_out[0]
        elif self.local_model:
            match local_mode_group[self.model_name][self.endpoint]:
                case 1:  # Uses chat template
                    pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the client for Granite model inference.

        For API mode, sets a placeholder client value since Replicate uses
        direct function calls rather than a persistent client. For GPU/CPU
        modes, validates the model is in local_mode_group but does not yet
        initialize a local model.

        Raises:
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            self.client = "replicate doesnt have a client"
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
                pass
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))

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

        Currently a no-op as Replicate uses stateless function calls.
        """
        pass