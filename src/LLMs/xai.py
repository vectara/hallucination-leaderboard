import os
from typing import Literal
from xai_sdk import Client
from xai_sdk.chat import user, system

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "xai-org"
class XAIConfig(BasicLLMConfig):
    """Extended config for x-ai-specific properties"""
    company: Literal["xai-org"] = "xai-org" 
    model_name: Literal[
        "grok-4.1",
        "grok-4.1-thinking",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
        "grok-4",
        "grok-3",
        "grok-3-mini",
        "grok-3-fast",
        "grok-3-mini-fast",
        "grok-2-vision"
    ] # Only model names manually added to this list are supported.
    date_code: str = "" # do we need date code?
    execution_mode: Literal["api"] = "api" # only API based?
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().
    reasoning_effort: Literal["NA", "low", "high"] = "NA"

class XAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class XAILLM(AbstractLLM):
    """
    Class for models from x-ai
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "grok-4.1":{
            "chat": 1
        },
        "grok-4.1-thinking":{
            "chat": 1
        },
        "grok-4-fast-non-reasoning":{
            "chat": 1
        },
        "grok-4-fast-reasoning":{
            "chat": 1
        },
        "grok-4-fast-non-reasoning":{
            "chat": 1
        },
        "grok-4":{
            "chat": 1
        },
        "grok-3":{
            "chat": 2
        },
        "grok-3-mini":{
            "chat": 2
        },
        "grok-3-fast":{
            "chat": 2
        },
        "grok-3-mini-fast":{
            "chat": 2
        },
        "grok-2-vision":{
            "chat": 2
        },
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: XAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.reasoning_effort = config.reasoning_effort
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Reasoning Model
                    chat = self.client.chat.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    chat.append(user(prepared_text))

                    response = chat.sample()
                    summary = response.content
                    self.thinking_tokens = response.usage.reasoning_tokens
                case 2: # Non reasoning models
                    chat = self.client.chat.create(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    chat.append(user(prepared_text))

                    response = chat.sample()
                    summary = response.content
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
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"XAI_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = Client(
                    api_host="api.x.ai",
                    api_key=api_key
                )
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
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass