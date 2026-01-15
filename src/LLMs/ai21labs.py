"""AI21 Labs (Jamba) model implementations for hallucination evaluation.

This module provides the LLM implementation for AI21 Labs' Jamba model family,
supporting API-based inference via the official AI21 Python client.

Classes:
    AI21LabsConfig: Configuration model for Jamba model settings.
    AI21LabsSummary: Output model for Jamba summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes (currently unused).
    AI21LabsLLM: Main LLM class implementing AbstractLLM for Jamba models.

Attributes:
    COMPANY: Provider identifier string ("ai21labs").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
import requests

from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "ai21labs"
"""str: Provider identifier used for API key lookup and model registration."""


class AI21LabsConfig(BasicLLMConfig):
    """Configuration model for AI21 Labs Jamba models.

    Extends BasicLLMConfig with AI21-specific settings for model selection
    and API configuration.

    Attributes:
        company: Provider identifier, fixed to "ai21labs".
        model_name: Name of the Jamba model variant to use. Some older
            versions (1.6) are deprecated.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["ai21labs"] = "ai21labs"
    model_name: Literal[
        "AI21-Jamba-Mini-1.5",
        "jamba-mini-2",
        "jamba-large-1.7",
        "jamba-mini-1.7",
        "jamba-large-1.6", # Deprecated
        "jamba-mini-1.6", # Deprecated
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

    class Config:
        """Pydantic configuration to forbid extra fields during parsing."""

        extra = "forbid"

class AI21LabsSummary(BasicSummary):
    """Output model for AI21 Labs Jamba summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for AI21 API client inference.

    Defines how the model should be invoked when using the AI21 API.

    Attributes:
        CHAT_DEFAULT: Use the official AI21 Python client for chat completions.
        CHAT_HTTP: Use raw HTTP requests to the AI21 REST API.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    CHAT_HTTP = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as AI21 models only support API inference.

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
# invoke the AI21 API.
client_mode_group = {
    "jamba-mini-2": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "jamba-large-1.7": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "jamba-mini-1.7": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "jamba-large-1.6": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "jamba-mini-1.6": {
        "chat": ClientMode.CHAT_DEFAULT
    },
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates AI21 models do not support local GPU execution.
local_mode_group = {}

class AI21LabsLLM(AbstractLLM):
    """LLM implementation for AI21 Labs Jamba models.

    Provides text summarization using AI21 Labs' Jamba model family via the
    official AI21 Python client or direct HTTP requests. Supports chat-formatted
    generation with configurable model variants.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
    """

    def __init__(self, config: AI21LabsConfig):
        """Initialize the AI21 Labs LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        # Call parent constructor to inherit all parent properties
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Jamba model via the AI21 API to generate a
        condensed summary. Supports both the official Python client and
        raw HTTP request modes.

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
                    messages = [
                        ChatMessage(role="user", content=prepared_text),
                    ]
                    response = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_fullname,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )

                    summary = response.choices[0].message.content
                case ClientMode.CHAT_HTTP:
                    api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                    url = "https://api.ai21.com/studio/v1/chat/completions"

                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }

                    payload = {
                        "model": self.model_fullname,
                        "messages": [
                            {"role": "user", "content": prepared_text}
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }

                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()

                    data = response.json()
                    summary = data["choices"][0]["message"]["content"]

        elif self.local_model: 
            pass 
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        # elif self.local_model_is_defined():
        #     if False:
        #         pass
        #     else:
        #         raise LocalModelProtocolBranchNotFound(self.model_name)
        # else:
        #     raise ClientOrLocalNotInitializedError(self.model_name)
        return summary

    def setup(self):
        """Initialize the AI21 API client for inference.

        Creates an AI21Client instance using the API key from the
        AI21LABS_API_KEY environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = AI21Client(api_key=api_key)
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
        """Close the AI21 API client connection.

        Currently a no-op as the AI21 client does not require explicit cleanup.
        """
        pass