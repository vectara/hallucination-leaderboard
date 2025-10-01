import os
from typing import Literal

from openai import OpenAI
from transformers import pipeline
from together import Together

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "openai"

class OpenAIConfig(BasicLLMConfig):
    """Extended config for OpenAI-specific properties"""
    company: Literal["openai"]
    model_name: Literal[
        "chatgpt-4o",
        "gpt-4.5-preview",
        "o1-preview",

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
    ] # Only model names manually added to this list are supported.
    execution_mode: Literal["api", "cpu", "gpu"] = "api" # OpenAI models can only be run via web api.
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = None

class OpenAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class OpenAILLM(AbstractLLM):
    """
    Class for models from OpenAI

    Attributes:
        client (OpenAI): client associated with api calls
        model_name (str): OpenAI style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    # Mode 1: Chat with temperature (default)
    # Mode 2: Chat without temperature
    # Mode 3: Use OpenAI's Response API
    client_mode_group = {
        "gpt-5-minimal": {
            "chat": 10,
            "api_type": "openai"
        },
        "gpt-5-high": {
            "chat": 11,
            "api_type": "openai"
        },
        "gpt-5": {
            "chat": 9,
            "api_type": "openai"
        },
        "gpt-5-mini": {
            "chat": 9,
            "api_type": "openai"
        },
        "gpt-5-nano": {
            "chat": 9,
            "api_type": "openai",
        },
        "gpt-4.1": {
            "chat": 1,
            "api_type": "openai",
            "response": 3
        },
        "gpt-4.1-nano": {
            "chat": 1,
            "api_type": "openai",
            "response": 3
        },
        "o3": {  # o3 does not support temperature
            "chat": 2,
            "api_type": "openai",
            "response": 3
        },
        "o3-pro": { # o3-pro doesn't support chatting protocol or temperature
            "chat": None, 
            "api_type": "openai",
            "response": 4
        },
        "o4-mini": {
            "chat": 2,
            "api_type": "openai",
        },
        "o4-mini-low": {
            "chat": 6,
            "api_type": "openai",
        },
        "o4-mini-high": {
            "chat": 7,
            "api_type": "openai",
        },
        "o1-pro": { # doesn't support chatting or temperature
            "chat": None,
            "api_type": "openai",
            "response": 4
        },
        "gpt-4.1-mini": {
            "chat": 1,
            "api_type": "openai",
        },
        "o1": {
            "chat": 2,
            "api_type": "openai",
        },
        "o1-mini": { # Doesn't support reasoning effort or temperature
            "chat": 5,
            "api_type": "openai",
        },
        "gpt-oss-120b": {
            "chat": 8,
            "api_type": "together"
        },
        "gpt-4o-mini": {
            "chat": 1,
            "api_type": "openai",
        },
        "gpt-4o": {
            "chat": 1,
            "api_type": "openai",
        },
        "gpt-4-turbo": {
            "chat": 1,
            "api_type": "openai",
        },
        "gpt-3.5-turbo": {
            "chat": 1,
            "api_type": "openai",
        },
        "gpt-4": {
            "chat": 1,
            "api_type": "openai",
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {
        "gpt-oss-20b": {
            "chat": 1
        },
    }

    def __init__(self, config: OpenAIConfig):

        # Call parent constructor to inherit all parent properties
        super().__init__(config)

        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.reasoning_effort = config.reasoning_effort
        if self.model_name in self.local_mode_group:
            self.model_fullname = f"openai/{self.model_fullname}"

        # Set default values for optional attributes
        # self.endpoint = config.endpoint if config.endpoint is not None else "chat" 
        # self.execution_mode = config.execution_mode if config.execution_mode is not None else "api"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Chat with temperature and max_tokens
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content":prepared_text}]
                    )
                    summary = chat_package.choices[0].message.content
                case 2: # Chat without temperature
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = self.reasoning_effort
                    )
                    summary = chat_package.choices[0].message.content
                case 6: # o4-mini-low
                    chat_package = self.client.chat.completions.create(
                        model="o4-mini-2025-04-16", # need to talk about this case
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = "low"
                    )
                    summary = chat_package.choices[0].message.content
                case 7: # o4-mini-high
                    chat_package = self.client.chat.completions.create(
                        model="o4-mini-2025-04-16", # need to talk about this case
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = "high"
                    )
                    summary = chat_package.choices[0].message.content
                case 9: #gpt-5
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
                case 10: # gpt-5-minimal
                    chat_package = self.client.responses.create(
                        model="gpt-5-2025-08-07", # need to talk about this case
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": self.reasoning_effort
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case 11: # gpt-5-high
                    chat_package = self.client.responses.create(
                        model="gpt-5-2025-08-07", # need to talk about this case
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": self.reasoning_effort
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case 8: # gpt-oss-120b not supported on open ai and too big to run locally, using together
                    together_name = f"openai/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                        model=together_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens = self.max_tokens,
                        temperature = self.temperature
                    )
                    summary = response.choices[0].message.content
                case 3: # Use OpenAI's Response API
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text,
                        reasoning = {"effort": self.reasoning_effort}
                    )
                    summary = chat_package.output_text
                case 4: # Use OpenAI's Response API no temp
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text,
                        reasoning = {"effort": self.reasoning_effort}
                    )
                    summary = chat_package.output_text
                case 5: # Chat without temperature and reasoning effort
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                    )
                    summary = chat_package.choices[0].message.content
                case None:
                    raise Exception(f"Model `{self.model_name}` cannot be run from `{self.endpoint}` endpoint")
        elif self.local_model:
            match self.local_mode_group[self.model_name][self.endpoint]:
                case 1: # Chat with temperature and max_tokens
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
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                if self.client_mode_group[self.model_name]["api_type"] == "together":
                    api_key = os.getenv(f"TOGETHER_API_KEY")
                    assert api_key is not None, f"TOGETHER API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = Together(api_key=api_key)
                else:
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
            if self.model_name in self.local_mode_group:
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
        if self.client:
            self.close_client()
        elif self.local_model:
            pass
            # self.default_local_model_teardown()

    def close_client(self):
        pass
