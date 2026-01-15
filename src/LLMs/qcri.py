"""QCRI (Fanar) model implementations for hallucination evaluation.

This module provides the LLM implementation for Qatar Computing Research
Institute's Fanar model family, supporting API-based inference via the
Fanar platform using an OpenAI-compatible client.

Classes:
    QCRIConfig: Configuration model for QCRI/Fanar model settings.
    QCRISummary: Output model for QCRI summarization results.
    QCRIJudgment: Output model for QCRI judgment results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    QCRILLM: Main LLM class implementing AbstractLLM for QCRI models.

Attributes:
    COMPANY: Provider identifier string ("qcri").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from openai import OpenAI

COMPANY = "qcri"
"""str: Provider identifier used for API key lookup and model registration."""


class QCRIConfig(BasicLLMConfig):
    """Configuration model for QCRI Fanar models.

    Extends BasicLLMConfig with QCRI/Fanar-specific settings for model
    selection and API configuration. Supports the Fanar model via the
    Fanar platform's OpenAI-compatible API.

    Attributes:
        company: Provider identifier, fixed to "qcri".
        model_name: Name of the Fanar model variant to use.
        execution_mode: Where to run inference, currently only "api" supported.
        date_code: Required version/date identifier for the model.
    """

    company: Literal["qcri"] = "qcri"
    model_name: Literal["fanar-model"]
    execution_mode: Literal["api"] = "api"
    date_code: str

class QCRISummary(BasicSummary):
    """Output model for QCRI summarization results.

    Inherits all fields from BasicSummary without additional attributes.
    Used for type consistency in QCRI model outputs.
    """

    pass


class QCRIJudgment(BasicJudgment):
    """Output model for QCRI judgment results.

    Inherits all fields from BasicJudgment without additional attributes.
    Used for type consistency in QCRI model judgment outputs.
    """

    pass

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the Fanar platform
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
    Currently unused as QCRI models are accessed via API only.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Maps directly to ClientMode values for the Fanar model.
client_mode_group = {
    "Fanar": ClientMode.CHAT_DEFAULT
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates QCRI models are accessed via API only.
local_mode_group = {}

class QCRILLM(AbstractLLM):
    """LLM implementation for QCRI Fanar models.

    Provides text summarization using Qatar Computing Research Institute's
    Fanar model via an OpenAI-compatible API hosted at api.fanar.qa.

    Attributes:
        Inherits all attributes from AbstractLLM.
    """

    def __init__(self, config: QCRIConfig):
        """Initialize the QCRI LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Fanar model via the QCRI API to generate a
        condensed summary. Sends the text as a user message to the chat
        completions endpoint.

        Args:
            prepared_text: The preprocessed text to summarize.

        Returns:
            The generated summary text, or an error placeholder if generation fails.

        Raises:
            Exception: If neither client nor local_model is initialized.
        """
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name]:
                case ClientMode.CHAT_DEFAULT:
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content":prepared_text}]
                    )
                    summary = chat_package.choices[0].message.content
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the Fanar API client for QCRI model inference.

        Creates an OpenAI-compatible client instance configured for the
        Fanar platform (api.fanar.qa) using the API key from the QCRI_API_KEY
        environment variable.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"Fanar API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = OpenAI(
                    base_url="https://api.fanar.qa/v1",
                    api_key=api_key
                )
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
        """Close the Fanar API client connection.

        Currently a no-op as the OpenAI-compatible client does not require
        explicit cleanup.
        """
        pass
