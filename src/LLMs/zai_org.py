"""Zhipu AI (GLM) model implementations for hallucination evaluation.

This module provides the LLM implementation for Zhipu AI's GLM model family,
supporting API-based inference via multiple backends including Together AI,
Fireworks AI, and DeepInfra. Includes support for GLM-4.5, GLM-4.6, and
GLM-4.7 series models.

Classes:
    ZhipuAIConfig: Configuration model for Zhipu AI model settings.
    ZhipuAISummary: Output model for Zhipu AI summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    ZhipuAILLM: Main LLM class implementing AbstractLLM for Zhipu AI models.

Attributes:
    COMPANY: Provider identifier string ("zai-org").
    client_mode_group: Mapping of models to supported API client modes and backends.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from together import Together
from openai import OpenAI
from enum import Enum, auto


from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
from huggingface_hub import InferenceClient

COMPANY = "zai-org"
"""str: Provider identifier used for model path construction and registration."""


class ZhipuAIConfig(BasicLLMConfig):
    """Configuration model for Zhipu AI GLM models.

    Extends BasicLLMConfig with Zhipu AI-specific settings for model selection
    and API configuration. Supports GLM models via multiple backend providers
    including Together AI, Fireworks AI, and DeepInfra.

    Attributes:
        company: Provider identifier, fixed to "zai-org".
        model_name: Name of the GLM model variant to use. Includes GLM-4.5
            (via Together), glm-4p5/4p7 (via Fireworks), GLM-4.6 (via DeepInfra),
            and glm-4-9b-chat for local execution.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["zai-org"] = "zai-org"
    model_name: Literal[
        "GLM-4.5-AIR-FP8",  # Together
        "glm-4p5",  # Fireworks but using OpenAI
        "glm-4p7",  # Fireworks but using OpenAI
        "glm-4-9b-chat",
        "GLM-4.6",
        "GLM-4.7",
        "GLM-4.7-Flash"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class ZhipuAISummary(BasicSummary):
    """Output model for Zhipu AI summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the various backend
    providers (Together AI, Fireworks AI, DeepInfra).

    Attributes:
        CHAT_DEFAULT: Standard chat completion mode.
        GLM_4P5_AIR_FP8: GLM-4.5-AIR-FP8 via Together AI.
        GLM_4P5: glm-4p5 via Fireworks AI with OpenAI-compatible client.
        GLM_4P6: GLM-4.6 via DeepInfra with OpenAI-compatible client.
        GLM_4P7: glm-4p7 via Fireworks AI with OpenAI-compatible client.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    GLM_4P5_AIR_FP8 = auto()
    GLM_4P5 = auto()
    GLM_4P6 = auto()
    GLM_4P7 = auto()
    GLM_4P7_FLASH = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Zhipu AI models are accessed via API only.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values and includes an api_type
# field indicating the backend provider (together, fireworks, or deepinfra).
client_mode_group = {
    "GLM-4.5-AIR-FP8": {
        "chat": ClientMode.GLM_4P5_AIR_FP8,
        "api_type": "together"
    },
    "glm-4p5": {
        "chat": ClientMode.GLM_4P5,
        "api_type": "fireworks"
    },
    "GLM-4.7-Flash": {
        "chat": ClientMode.GLM_4P7_FLASH,
        "api_type": "huggingface"
    },
    "glm-4p7": {
        "chat": ClientMode.GLM_4P7,
        "api_type": "fireworks"
    },
    "GLM-4.6": {
        "chat": ClientMode.GLM_4P6,
        "api_type": "deepinfra"
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Zhipu AI models are accessed via API only.
local_mode_group = {}

class ZhipuAILLM(AbstractLLM):
    """LLM implementation for Zhipu AI GLM models.

    Provides text summarization using Zhipu AI's GLM model family via multiple
    backend providers. Supports Together AI for GLM-4.5, Fireworks AI for
    glm-4p5/4p7 variants, and DeepInfra for GLM-4.6. Each backend uses either
    the Together SDK or an OpenAI-compatible client.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        full_config: Complete configuration object for reference.
    """

    def __init__(self, config: ZhipuAIConfig):
        """Initialize the Zhipu AI LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured GLM model via the appropriate backend provider
        to generate a condensed summary. Routes requests to Together AI,
        Fireworks AI, or DeepInfra based on the model's api_type configuration.

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
                case ClientMode.GLM_4P5_AIR_FP8:
                    together_name = f"zai-org/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                        model=together_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens = self.max_tokens,
                        temperature = self.temperature
                    )
                    summary = response.choices[0].message.content
                case ClientMode.GLM_4P5:
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
                case ClientMode.GLM_4P7:
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

                case ClientMode.GLM_4P7_FLASH:
                    messages = [{"role": "user", "content": prepared_text}]
                    client_package = self.client.chat_completion(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = client_package.choices[0].message.content

                case ClientMode.GLM_4P6:
                    self.model_fullname = f"{COMPANY}/{self.model_name}"
                    chat_completion = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content": prepared_text}],
                    )
                    summary = chat_completion.choices[0].message.content

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
        """Initialize the appropriate API client for model inference.

        Creates a client instance based on the model's api_type configuration:
        - Together AI: Uses the Together SDK with TOGETHER_API_KEY
        - Fireworks AI: Uses OpenAI-compatible client with FIREWORKS_API_KEY
        - DeepInfra: Uses OpenAI-compatible client with DEEPINFRA_API_KEY

        Raises:
            AssertionError: If the required API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if client_mode_group[self.model_name]["api_type"] == "together":
                    api_key = os.getenv(f"TOGETHER_API_KEY")
                    assert api_key is not None, f"TOGETHER API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = Together(api_key=api_key)
                elif client_mode_group[self.model_name]["api_type"] == "fireworks":
                    api_key = os.getenv(f"FIREWORKS_API_KEY")
                    assert api_key is not None, f"FIREWORKS API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.fireworks.ai/inference/v1"
                    )
                elif client_mode_group[self.model_name]["api_type"] == "deepinfra":
                    api_key = os.getenv(f"DEEPINFRA_API_KEY")
                    assert api_key is not None, f"DEEPINFRA API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepinfra.com/v1/openai"
                    )
                elif client_mode_group[self.model_name]["api_type"] == "huggingface":
                    self.model_fullname = f"{COMPANY}/{self.model_name}"
                    self.client = InferenceClient(model=self.model_fullname)
                
                else:
                    self.client  = None
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
        """Close the API client connection.

        Currently a no-op as the Together SDK and OpenAI-compatible clients
        do not require explicit cleanup.
        """
        pass