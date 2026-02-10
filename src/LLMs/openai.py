"""OpenAI model implementations for hallucination evaluation.

This module provides the LLM implementation for OpenAI's model family,
supporting API-based inference via the official OpenAI SDK, Together AI,
and Replicate platforms. Also supports local execution via HuggingFace
transformers for open-source variants. Includes GPT-3.5, GPT-4, GPT-4o,
GPT-5 series, and reasoning models (o1, o3, o4).

Classes:
    OpenAIConfig: Configuration model for OpenAI model settings.
    OpenAISummary: Output model for OpenAI summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    OpenAILLM: Main LLM class implementing AbstractLLM for OpenAI models.

Attributes:
    COMPANY: Provider identifier string ("openai").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes.
"""

import os
from typing import Literal
from enum import Enum, auto

from openai import OpenAI
from transformers import pipeline
from together import Together
import replicate

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "openai"
"""str: Provider identifier used for API key lookup and model registration."""


class OpenAIConfig(BasicLLMConfig):
    """Configuration model for OpenAI models.

    Extends BasicLLMConfig with OpenAI-specific settings for model selection,
    execution mode, and reasoning effort configuration. Supports the full range
    of OpenAI models from GPT-3.5 through GPT-5, including reasoning models
    (o1, o3, o4 series) and open-source variants (gpt-oss).

    Attributes:
        company: Provider identifier, fixed to "openai".
        model_name: Name of the OpenAI model variant to use. Includes GPT-3.5/4/4o,
            GPT-4.1, GPT-5 series, reasoning models (o1/o3/o4), and gpt-oss variants.
        execution_mode: Where to run inference ("api", "cpu", or "gpu").
        endpoint: API endpoint type ("chat" for chat completions, "response" for
            responses API used by reasoning models).
        reasoning_effort: Reasoning intensity for supported models ("none", "minimal",
            "low", "medium", "high"). Used by o-series and GPT-5 models.
        api_type: Backend API to use. "default" for OpenAI, "together" for Together AI,
            "replicate" for Replicate. gpt-oss models require explicit api_type.
    """

    company: Literal["openai"] = "openai"
    model_name: Literal[
        "chatgpt-4o",
        "gpt-4.5-preview",
        "o1-preview",

        "gpt-5.2-high",
        "gpt-5.2-low",
        "gpt-5.1-high",
        "gpt-5.1-low",
        "gpt-5-high",
        "gpt-5-minimal",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-oss-120b",
        "gpt-oss-20b",
        "gpt-4.1",
        "gpt-4.1-nano",
        "o3",
        "o3-pro",
        "o4-mini",
        "o4-mini-low",
        "o4-mini-high",
        "o1-pro",
        "gpt-4.1-mini",
        "o1",
        "o1-mini",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-4"
    ]
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = None
    api_type: Literal["default", "together", "replicate"] = "default"

class OpenAISummary(BasicSummary):
    """Output model for OpenAI summarization results.

    Extends BasicSummary with endpoint, reasoning effort, and api_type tracking
    for result provenance. Captures configuration used during generation.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
        reasoning_effort: The reasoning effort level used, if applicable.
        api_type: The backend API used ("default" for OpenAI, "together", "replicate").
    """

    endpoint: Literal["chat", "response"] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    api_type: Literal["default", "together", "replicate"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using OpenAI, Together AI,
    or Replicate platforms. Different modes handle temperature and reasoning
    parameters appropriately for each model type.

    Attributes:
        CHAT_DEFAULT: Standard chat completion with temperature and max_tokens.
        CHAT_NO_TEMP: Chat completion without temperature (for reasoning models).
        CHAT_NO_TEMP_NO_REASONING: Chat completion without temperature or reasoning.
        RESPONSE_DEFAULT: Responses API with temperature and reasoning effort.
        RESPONSE_NO_TEMP: Responses API without temperature parameter.
        DEFAULT_TOGETHER_API: Chat completion via Together AI platform.
        DEFAULT_REPLICATE_API: Generation via Replicate platform.
        O4_MINI_LOW: O4-mini model with low reasoning effort.
        O4_MINI_HIGH: O4-mini model with high reasoning effort.
        GPT_5P2_HIGH: GPT-5.2 model with high reasoning effort.
        GPT_5P2_LOW: GPT-5.2 model with low reasoning effort.
        GPT_5P1_HIGH: GPT-5.1 model with high reasoning effort.
        GPT_5P1_LOW: GPT-5.1 model with low reasoning effort.
        GPT_5_HIGH: GPT-5 model with high reasoning effort.
        GPT_5_MINIMAL: GPT-5 model with minimal reasoning effort.
        GPT_5_DEFAULT: GPT-5 model with configurable reasoning effort.
    """

    CHAT_DEFAULT = auto()
    CHAT_NO_TEMP = auto()
    CHAT_NO_TEMP_NO_REASONING = auto()
    RESPONSE_DEFAULT = auto()
    RESPONSE_NO_TEMP = auto()
    DEFAULT_TOGETHER_API = auto()
    DEFAULT_REPLICATE_API = auto()
    O4_MINI_LOW = auto()
    O4_MINI_HIGH = auto()
    GPT_5P2_HIGH = auto()
    GPT_5P2_LOW = auto()
    GPT_5P1_HIGH = auto()
    GPT_5P1_LOW = auto()
    GPT_5_HIGH = auto()
    GPT_5_MINIMAL = auto()
    GPT_5_DEFAULT = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally via
    HuggingFace transformers pipeline.

    Attributes:
        CHAT_DEFAULT: Use text-generation pipeline with chat message formatting.
    """

    CHAT_DEFAULT = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values.
# Models may support chat, response, or both endpoints.
client_mode_group = {
    "gpt-5.2-low": {"chat": ClientMode.GPT_5P2_LOW},
    "gpt-5.2-high": {"chat": ClientMode.GPT_5P2_HIGH},
    "gpt-5.1-low": {"chat": ClientMode.GPT_5P1_LOW},
    "gpt-5.1-high": {"chat": ClientMode.GPT_5P1_HIGH},
    "gpt-5-minimal": {"chat": ClientMode.GPT_5_MINIMAL},
    "gpt-5-high": {"chat": ClientMode.GPT_5_HIGH},
    "gpt-5": {"chat": ClientMode.GPT_5_DEFAULT},
    "gpt-5-mini": {"chat": ClientMode.GPT_5_DEFAULT},
    "gpt-5-nano": {"chat": ClientMode.GPT_5_DEFAULT},
    "gpt-4.1": {"chat": ClientMode.CHAT_DEFAULT, "response": ClientMode.RESPONSE_DEFAULT},
    "gpt-4.1-nano": {"chat": ClientMode.CHAT_DEFAULT, "response": ClientMode.RESPONSE_DEFAULT},
    "o3": {"chat": ClientMode.CHAT_NO_TEMP, "response": ClientMode.RESPONSE_DEFAULT},
    "o3-pro": {"response": ClientMode.RESPONSE_NO_TEMP},
    "o4-mini": {"chat": ClientMode.CHAT_NO_TEMP},
    "o4-mini-low": {"chat": ClientMode.O4_MINI_LOW},
    "o4-mini-high": {"chat": ClientMode.O4_MINI_HIGH},
    "o1-pro": {"response": ClientMode.RESPONSE_NO_TEMP},
    "gpt-4.1-mini": {"chat": ClientMode.CHAT_DEFAULT},
    "o1": {"chat": ClientMode.CHAT_NO_TEMP},
    "o1-mini": {"chat": ClientMode.CHAT_NO_TEMP_NO_REASONING},
    "gpt-oss-120b": {"chat": ClientMode.DEFAULT_TOGETHER_API},
    "gpt-oss-20b": {"chat": ClientMode.DEFAULT_REPLICATE_API},
    "gpt-4o-mini": {"chat": ClientMode.CHAT_DEFAULT},
    "gpt-4o": {"chat": ClientMode.CHAT_DEFAULT},
    "gpt-4-turbo": {"chat": ClientMode.CHAT_DEFAULT},
    "gpt-3.5-turbo": {"chat": ClientMode.CHAT_DEFAULT},
    "gpt-4": {"chat": ClientMode.CHAT_DEFAULT},
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Contains open-source model variants that can be run locally via transformers.
local_mode_group = {
    "gpt-oss-20b": {
        "chat": LocalMode.CHAT_DEFAULT
    },
}


class OpenAILLM(AbstractLLM):
    """LLM implementation for OpenAI models.

    Provides text summarization using OpenAI's model family via multiple
    backends: official OpenAI SDK, Together AI, Replicate, and local
    execution via HuggingFace transformers. Supports chat completions
    and responses API with configurable reasoning effort for advanced models.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat" or "response").
        execution_mode: Where inference runs ("api", "cpu", or "gpu").
        reasoning_effort: Reasoning intensity for o-series and GPT-5 models.
    """

    def __init__(self, config: OpenAIConfig):
        """Initialize the OpenAI LLM with the given configuration.

        Args:
            config: Configuration object specifying model and execution settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.reasoning_effort = config.reasoning_effort
        self.api_type = config.api_type
        if self.model_name in local_mode_group:
            self.model_fullname = f"openai/{self.model_fullname}"

    def extract_summary(self, resp):
        """Extract text content from a responses API response object.

        Parses the structured response format from OpenAI's responses API
        to find and return the generated text content.

        Args:
            resp: Response object from OpenAI responses API.

        Returns:
            The extracted text content, or empty string if not found.
        """
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                if getattr(item, "content", None):
                    for c in item.content:
                        if getattr(c, "text", None):
                            return c.text
        return ""

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured OpenAI model via the appropriate API backend to
        generate a condensed summary. Routes to chat completions, responses API,
        Together AI, Replicate, or local inference based on model configuration.

        Args:
            prepared_text: The preprocessed text to summarize.

        Returns:
            The generated summary text, or an error placeholder if generation fails.

        Raises:
            Exception: If neither client nor local_model is initialized, or if
                the model cannot be run from the configured endpoint.
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
                case ClientMode.CHAT_NO_TEMP:
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = self.reasoning_effort
                    )
                    summary = chat_package.choices[0].message.content
                case ClientMode.O4_MINI_LOW:
                    chat_package = self.client.chat.completions.create(
                        model="o4-mini-2025-04-16",
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = "low"
                    )
                    summary = chat_package.choices[0].message.content
                case ClientMode.O4_MINI_HIGH:
                    chat_package = self.client.chat.completions.create(
                        model="o4-mini-2025-04-16",
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = "high"
                    )
                    summary = chat_package.choices[0].message.content
                case ClientMode.GPT_5_DEFAULT:
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": self.reasoning_effort
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5_MINIMAL:
                    chat_package = self.client.responses.create(
                        model="gpt-5-2025-08-07",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "minimal"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5_HIGH:
                    chat_package = self.client.responses.create(
                        model="gpt-5-2025-08-07",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5P1_LOW:
                    chat_package = self.client.responses.create(
                        model="gpt-5.1-2025-11-13",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "low"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5P1_HIGH:
                    chat_package = self.client.responses.create(
                        model="gpt-5.1-2025-11-13",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5P2_LOW:
                    chat_package = self.client.responses.create(
                        model="gpt-5.2-2025-12-11",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "low"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = self.extract_summary(chat_package)
                case ClientMode.GPT_5P2_HIGH:
                    chat_package = self.client.responses.create(
                        model="gpt-5.2-2025-12-11",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = self.extract_summary(chat_package)
                case ClientMode.DEFAULT_TOGETHER_API:
                    together_name = f"openai/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                        model=together_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens = self.max_tokens,
                        temperature = self.temperature
                    )
                    summary = response.choices[0].message.content
                case ClientMode.DEFAULT_REPLICATE_API:
                    input = {
                        "prompt": prepared_text,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                    }
                    summary = replicate.run(
                        f"{COMPANY}/{self.model_name}",
                        input=input
                    )
                    summary = summary[0]
                case ClientMode.RESPONSE_DEFAULT:
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text,
                        reasoning = {"effort": self.reasoning_effort}
                    )
                    summary = chat_package.output_text
                case ClientMode.RESPONSE_NO_TEMP:
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text,
                        reasoning = {"effort": self.reasoning_effort}
                    )
                    summary = chat_package.output_text
                case ClientMode.CHAT_NO_TEMP_NO_REASONING:
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                    )
                    summary = chat_package.choices[0].message.content
                case None:
                    raise Exception(f"Model `{self.model_name}` cannot be run from `{self.endpoint}` endpoint")
        elif self.local_model:
            match local_mode_group[self.model_name][self.endpoint]:
                case LocalMode.CHAT_DEFAULT:
                    def extract_after_assistant_final(text):
                        keyword = "assistantfinal"
                        index = text.find(keyword)
                        if index != -1:
                            return text[index + len(keyword):].strip()
                        return ""  # Return empty string if keyword not found
                    messages = [
                        {"role": "user", "content": prepared_text},
                    ]

                    outputs = self.local_model(
                        messages,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    raw_text = outputs[0]["generated_text"][-1]["content"]
                    summary = extract_after_assistant_final(raw_text)
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the appropriate client for OpenAI model inference.

        Creates the API client based on the model's api_type configuration:
        - "openai": Official OpenAI SDK using OPENAI_API_KEY
        - "together": Together AI client using TOGETHER_API_KEY
        - "replicate": Replicate API using REPLICATE_API_TOKEN
        For local execution, initializes a HuggingFace transformers pipeline.

        Raises:
            AssertionError: If the required API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "together":
                    api_key = os.getenv(f"TOGETHER_API_KEY")
                    assert api_key is not None, f"TOGETHER API key not found in environment variable TOGETHER_API_KEY"
                    self.client = Together(api_key=api_key)
                elif self.api_type == "replicate":
                    api_key = os.getenv(f"REPLICATE_API_TOKEN")
                    assert api_key is not None, f"REPLICATE API key not found in environment variable REPLICATE_API_TOKEN"
                    self.client = "replicate has no client"
                else:  # default
                    api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                    assert api_key is not None, f"OpenAI API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(api_key=api_key)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
                self.local_model = pipeline(
                    "text-generation",
                    model=self.model_fullname,
                    torch_dtype="auto",
                    device_map="auto", # Set gpu?
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

        Currently a no-op as the OpenAI client does not require explicit cleanup.
        """
        pass
