"""Ant Group (Finix/AntFinix) model implementations for hallucination evaluation.

This module provides the LLM implementation for Ant Group's Finix and AntFinix
model family, supporting API-based inference via the AntFinix REST API with
streaming responses.

Classes:
    AntGroupMIConfig: Configuration model for AntFinix model settings.
    AntGroupMISummary: Output model for AntFinix summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes (currently unused).
    AntGroupMILLM: Main LLM class implementing AbstractLLM for AntFinix models.

Attributes:
    COMPANY: Provider identifier string ("antgroup").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

import json
import multiprocessing
import requests

COMPANY = "antgroup"
"""str: Provider identifier used for API key lookup and model registration."""

class AntGroupMIConfig(BasicLLMConfig):
    """Configuration model for Ant Group AntFinix models.

    Extends BasicLLMConfig with AntFinix-specific settings for model selection
    and API configuration.

    Attributes:
        company: Provider identifier, fixed to "antgroup".
        model_name: Name of the AntFinix model variant to use.
        execution_mode: Where to run inference, currently only "api" supported.
        date_code: Optional version/date identifier for the model.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["antgroup"] = "antgroup"
    model_name: Literal[
        "finix_s1_32b",
        "antfinix-a1"
    ]
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"

class AntGroupMISummary(BasicSummary):
    """Output model for Ant Group AntFinix summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for AntFinix API client inference.

    Defines how the model should be invoked when using the AntFinix REST API.
    Different models may use different API call implementations.

    Attributes:
        CHAT_DEFAULT: Standard chat completion endpoint.
        FINIX_S1_32B: Finix S1 32B model with streaming v1 API.
        ANTFINIX_A1: AntFinix A1 model with streaming v2 API (supports reasoning).
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    FINIX_S1_32B = auto()
    ANTFINIX_A1 = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as AntFinix models only support API inference.

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
# API call method to use (v1 or v2 streaming).
client_mode_group = {
    "finix_s1_32b": {
        "chat": ClientMode.FINIX_S1_32B
    },
    "antfinix-a1": {
        "chat": ClientMode.ANTFINIX_A1
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates AntFinix models do not support local execution.
local_mode_group = {}

class AntGroupMILLM(AbstractLLM):
    """LLM implementation for Ant Group AntFinix models.

    Provides text summarization using Ant Group's Finix and AntFinix model
    family via the AntFinix REST API. Uses streaming responses to handle
    long-running inference requests and avoid timeouts.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        api_key: API key for AntFinix authentication.
    """

    def __init__(self, config: AntGroupMIConfig):
        """Initialize the Ant Group LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured AntFinix model via the streaming API to generate
        a condensed summary. Routes to the appropriate API call method based
        on the model type.

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
                case ClientMode.FINIX_S1_32B:
                    base_url = "https://antfinix.alipay.com/v1/chat/completions"
                    messages = [{"role": "user", "content": prepared_text}]
                    summary = self.call_insllm_api(
                        api_token=self.api_key,
                        base_url=base_url,
                        model_name="antfinix-ir1",
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )

                case ClientMode.ANTFINIX_A1:
                    base_url = "https://antfinix.alipay.com/v1/chat/completions"
                    messages = [{"role": "user", "content": prepared_text}]
                    summary = self.call_insllm_api_v2(
                        api_token=self.api_key,
                        base_url=base_url,
                        model_name="antfinix-a1",
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    pass
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        """Initialize the API key for AntFinix inference.

        Loads the API key from the ANTGROUP_API_KEY environment variable
        and sets the client flag to indicate readiness.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                model_id = COMPANY.replace("-", "_")
                self.api_key = os.getenv(f"{model_id.upper()}_API_KEY")
                assert self.api_key is not None, f"{COMPANY} API key not found in environment variable {model_id}"
                self.client = True
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
        """Close the API client connection.

        Currently a no-op as the REST client does not require explicit cleanup.
        """
        pass

    def call_insllm_api(
            self,
            api_token,
            base_url,
            model_name,
            messages,
            temperature,
            max_tokens
        ):
        """Make a streaming API request to the AntFinix v1 endpoint.

        Sends a chat completion request with streaming enabled to avoid
        timeouts during long inference. Parses Server-Sent Events (SSE)
        format responses.

        Args:
            api_token: Bearer token for API authentication.
            base_url: The AntFinix API endpoint URL.
            model_name: The model identifier for the request.
            messages: List of chat messages in OpenAI format.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate.

        Returns:
            The complete generated text accumulated from stream chunks,
            or None if the request fails.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,  # avoid timeout for long think and response
        }
        try:
            response = requests.post(base_url, headers=headers, json=payload, timeout=600)
            completion = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode("utf-8")
                    if line_text.startswith("data: "):
                        json_str = line_text[6:]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                content = chunk["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    completion += content
                        except json.JSONDecodeError:
                            print(f"Error Decoding JSON: {json_str}")
            
            return completion
        except Exception as e:
            print(f"Error calling INSLLM API: {e}")
            return None

    def call_insllm_api_v2(
            self,
            api_token,
            base_url,
            model_name,
            messages,
            temperature,
            max_tokens
        ):
        """Make a streaming API request to the AntFinix v2 endpoint.

        Similar to call_insllm_api but handles the v2 response format which
        includes separate reasoning_content and content fields. Only content
        is accumulated; reasoning tokens are ignored.

        Args:
            api_token: Bearer token for API authentication.
            base_url: The AntFinix API endpoint URL.
            model_name: The model identifier for the request.
            messages: List of chat messages in OpenAI format.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate.

        Returns:
            The complete generated text accumulated from stream chunks,
            or an error message string if the request fails.
        """
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {api_token}"
        }
        stream_state = True
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream_state,  # avoid timeout for long think and response
        }
        try:
            response = requests.post(base_url, headers=headers, json=payload, stream=stream_state)
            result = ""
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        json_str = line[5:]
                        if json_str.strip() == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                                delta = chunk["choices"][0]["delta"]
                                if "reasoning_content" in delta and delta["reasoning_content"]:
                                    pass
                                elif "content" in delta and delta["content"]:
                                    result += delta["content"]

                        except json.JSONDecodeError:
                            print(f"Error Decoding JSON: {json_str}")
            
            return result
        except Exception as e:
            print(f"Error calling INSLLM API: {e}")
            return "ERROR CALLING INSLLM API"