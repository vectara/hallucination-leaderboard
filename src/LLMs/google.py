"""Google (Gemini, Gemma) model implementations for hallucination evaluation.

This module provides the LLM implementation for Google's Gemini and Gemma model
families, supporting multiple inference backends: Google's genai SDK for Gemini
API, Replicate for Gemma 3 models, and local HuggingFace transformers for
smaller Gemma variants.

Classes:
    GoogleConfig: Configuration model for Google model settings.
    GoogleSummary: Output model for Google summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    GoogleLLM: Main LLM class implementing AbstractLLM for Google models.

Attributes:
    COMPANY: Provider identifier string ("google").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to supported local execution modes.
"""

import os
from typing import Literal
from enum import Enum, auto

from google import genai
from google.genai import types
import torch

from .AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from transformers import pipeline

import replicate

COMPANY = "google"
"""str: Provider identifier used for API key lookup and model registration."""

class GoogleConfig(BasicLLMConfig):
    """Configuration model for Google Gemini and Gemma models.

    Extends BasicLLMConfig with Google-specific settings for model selection,
    execution mode, and thinking budget configuration. Supports Gemini API models,
    Gemma models via Replicate, and local Gemma inference.

    Attributes:
        company: Provider identifier, fixed to "google".
        model_name: Name of the Google model variant to use. Includes Gemini
            1.5/2.0/2.5/3 series and Gemma 1.x/2/3 instruction-tuned models.
        endpoint: API endpoint type ("chat" for conversational format).
        execution_mode: Where to run inference ("api", "gpu", or "cpu").
        date_code: Optional version/date identifier for the model.
        thinking_budget: Thinking token budget for Gemini 2.5 models.
            -1 enables dynamic thinking, 0 disables thinking.
    """

    company: Literal["google"] = "google"
    model_name: Literal[
        "chat-bison-001",
        "flan-t5-large",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro-001",
        "gemini-2.0-flash-lite-preview",
        "gemini-2.0-pro-exp",
        "gemini-2.5-pro-exp",
        "text-bison-001",

        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash-preview",  # 05-20
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemma-7b-it",  # Try local
        "gemma-1.1-2b-it",  # Try local
        "gemma-1.1-7b-it",  # Try local
        "gemma-2-2b-it",  # Try local
        "gemma-2-9b-it",  # Try local
        "google/flan-t5-large"  # Use through huggingface

    ]
    endpoint: Literal["chat", "response"] = "chat"
    execution_mode: Literal["api", "gpu", "cpu"] = "api"
    date_code: str = "",
    thinking_budget: Literal[-1, 0] = 0  # -1 is dynamic thinking, 0 thinking is off

class GoogleSummary(BasicSummary):
    """Output model for Google summarization results.

    Extends BasicSummary with endpoint and thinking budget tracking for
    result provenance and configuration audit.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
        thinking_budget: The thinking token budget used for Gemini 2.5 models.
            -1 indicates dynamic thinking, 0 indicates thinking was disabled.
    """

    endpoint: Literal["chat", "response"] | None = None
    thinking_budget: Literal[-1, 0] | None = None  # -1 is dynamic thinking, 0 thinking is off

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using various API clients.
    Supports Google's genai SDK and Replicate for different model families.

    Attributes:
        CHAT_DEFAULT: Standard Gemini API chat generation.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
        CHAT_THINKING_BUDGET: Gemini 2.5 API with thinking budget configuration.
        REPLICATE_GEMMA_27B_IT: Gemma 3 27B via Replicate API.
        REPLICATE_GEMMA_12B_IT: Gemma 3 12B via Replicate API.
        REPLICATE_GEMMA_4B_IT: Gemma 3 4B via Replicate API.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()
    CHAT_THINKING_BUDGET = auto()
    REPLICATE_GEMMA_27B_IT = auto()
    REPLICATE_GEMMA_12B_IT = auto()
    REPLICATE_GEMMA_4B_IT = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally via
    HuggingFace transformers pipeline.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting with text-generation pipeline.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values indicating how to
# invoke the API. Gemini 2.5 models use CHAT_THINKING_BUDGET for thinking
# configuration. Gemma 3 models use Replicate-specific modes.
client_mode_group = {
    "gemini-3-flash-preview": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-3-pro-preview": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-2.5-flash-preview": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemma-3-4b-it": {
        "chat": ClientMode.REPLICATE_GEMMA_4B_IT
    },
    "gemma-3-12b-it": {
        "chat": ClientMode.REPLICATE_GEMMA_12B_IT
    },
    "gemma-3-27b-it": {
        "chat": ClientMode.REPLICATE_GEMMA_27B_IT
    },
    "gemini-2.0-pro-exp": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-2.0-flash-001": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-2.0-flash": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-2.0-flash-exp": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-2.0-flash-lite": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-1.5-flash-002": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-1.5-pro-002": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-1.5-flash": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-1.5-pro": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-pro": {
        "chat": ClientMode.CHAT_DEFAULT
    },  # Prob does not work
    "gemma-7n-it": {
        "chat": ClientMode.CHAT_DEFAULT
    },  # Not officially listed
    "gemma-1.1-2b-it": {
        "chat": ClientMode.CHAT_DEFAULT
    },  # Not officially listed
    "gemma-1.1-7b-it": {
        "chat": ClientMode.CHAT_DEFAULT
    },  # Not officially listed
    "gemma-2-2b-it": {
        "chat": ClientMode.CHAT_DEFAULT
    },  # Not officially listed
    "gemma-2-9b-it": {
        "chat": ClientMode.CHAT_DEFAULT
    },  # Not officially listed
    "gemini-2.5-pro": {
        "chat": ClientMode.CHAT_THINKING_BUDGET
    },
    "gemini-2.5-pro-preview": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "gemini-2.5-flash-lite": {
        "chat": ClientMode.CHAT_THINKING_BUDGET
    },
    "gemini-2.5-flash": {
        "chat": ClientMode.CHAT_THINKING_BUDGET
    },
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Smaller Gemma 3 models support local inference via HuggingFace transformers.
local_mode_group = {
    "gemma-3-1b-it": {
        "chat": LocalMode.CHAT_DEFAULT
    },
    "gemma-3-4b-it": {
        "chat": LocalMode.CHAT_DEFAULT
    },
    "gemma-3-12b-it": {
        "chat": LocalMode.CHAT_DEFAULT
    },
}

class GoogleLLM(AbstractLLM):
    """LLM implementation for Google Gemini and Gemma models.

    Provides text summarization using Google's Gemini and Gemma model families
    via multiple backends: Google's genai SDK for Gemini API, Replicate for
    larger Gemma 3 models, and local HuggingFace transformers for smaller
    Gemma variants. Supports thinking budget configuration for Gemini 2.5.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs ("api", "gpu", or "cpu").
        thinking_budget: Thinking token budget for Gemini 2.5 models.
        model_fullname: Full HuggingFace model path for local models.
    """

    def __init__(self, config: GoogleConfig):
        """Initialize the Google LLM with the given configuration.

        Args:
            config: Configuration object specifying model and execution settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.thinking_budget = config.thinking_budget
        if self.model_name in local_mode_group:
            self.model_fullname = f"{COMPANY}/{self.model_name}"


    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Google model to generate a condensed summary.
        Routes to the appropriate backend based on model type:
        - CHAT_DEFAULT: Standard Gemini API generation
        - CHAT_THINKING_BUDGET: Gemini 2.5 with thinking configuration
        - REPLICATE_GEMMA_*: Gemma 3 models via Replicate API
        - LocalMode.CHAT_DEFAULT: Local HuggingFace pipeline

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
                    response = self.client.models.generate_content(
                        model=self.model_fullname,
                        contents=prepared_text,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        ),
                    )
                    summary = response.text
                case ClientMode.CHAT_THINKING_BUDGET:
                    response = self.client.models.generate_content(
                        model=self.model_fullname,
                        contents=prepared_text,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_tokens)
                        ),
                    )
                    summary = response.text
                case ClientMode.REPLICATE_GEMMA_27B_IT:
                    input = {
                        "prompt": prepared_text,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                    }
                    summary = replicate.run(
                        f"google-deepmind/{self.model_name}:c0f0aebe8e578c15a7531e08a62cf01206f5870e9d0a67804b8152822db58c54",
                        input=input
                    )
                    summary = summary.replace("<end_of_turn>", "")
                case ClientMode.REPLICATE_GEMMA_12B_IT:
                    input = {
                        "prompt": prepared_text,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                    }
                    summary = replicate.run(
                        f"google-deepmind/{self.model_name}:5a0df3fa58c87fbd925469a673fdb16f3dd08e6f4e2f1a010970f07b7067a81c",
                        input=input
                    )
                    summary = summary.replace("<end_of_turn>", "")
                case ClientMode.REPLICATE_GEMMA_4B_IT:
                    input = {
                        "prompt": prepared_text,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                    }
                    summary = replicate.run(
                        f"google-deepmind/{self.model_name}:00139d2960396352b671f7b5c2ece5313bf6d45fe0a052efe14f023d2a81e196",
                        input=input
                    )
                    summary = summary.replace("<end_of_turn>", "")
        elif self.local_model:
            match local_mode_group[self.model_name][self.endpoint]:
                case LocalMode.CHAT_DEFAULT:  # Uses chat template
                    print("ATTEMPTING TO REQUEST")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prepared_text}
                            ]
                        }
                    ]

                    output = self.local_model(
                        text=messages,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    summary = output[0]["generated_text"][-1]["content"]
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the appropriate client for Google model inference.

        For API mode, creates a Google genai Client using the API key from
        the GOOGLE_GEMINI_API_KEY environment variable. For GPU/CPU modes,
        creates a HuggingFace text-generation pipeline with bfloat16 precision.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_GEMINI_API_KEY")
                assert api_key is not None, f"Google Gemini API key not found in environment variable {COMPANY.upper()}_GEMINI_API_KEY"
                self.client = genai.Client(api_key=api_key)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
                self.local_mode = pipeline(
                    "text-generation",
                    model=self.model_fullname,
                    device="cuda",
                    torch_dtype=torch.bfloat16
                )
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

        Currently a no-op as the Google genai client does not require
        explicit cleanup.
        """
        pass
