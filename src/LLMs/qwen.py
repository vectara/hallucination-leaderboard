import os
from typing import Literal

from openai import OpenAI

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

"""
Unique Notes:

qwen-max
"""

#TODO: Rename Qwen

COMPANY = "qwen" # Previously alibaba
class QwenConfig(BasicLLMConfig):
    """Extended config for Alibaba-specific properties"""
    company: Literal["qwen"] 
    model_name: Literal[
        "qwen3-max-preview",
        "qwen3-32b",
        "qwen3-14b",
        "qwen3-8b",
        "qwen3-4b",
        "qwen3-1.7b",
        "qwen3-0.6b",
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "qwen2.5-72b-instruct", 
        "qwen2.5-32b-instruct", 
        "qwen2.5-14b-instruct", 
        "qwen2.5-7b-instruct", 
    ] # Only model names manually added to this list are supported.
    date_code: str = "" # do we need date code for ali baba?
    execution_mode: Literal["api"] = "api" # Is Alibaba only API based?
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().
    thinking_tokens: bool = None
    enable_thinking: bool = None

class QwenSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.
    enable_thinking: bool | None = None

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class QwenLLM(AbstractLLM):
    """
    Class for models from Alibaba
    """
    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "qwen3-max-preview": {
            "chat": 2
        },
        "qwen3-32b": {
            "chat": 2
        },
        "qwen3-14b": {
            "chat": 2
        },
        "qwen3-8b": {
            "chat": 2
        },
        "qwen3-4b": {
            "chat": 2
        },
        "qwen3-1.7b": {
            "chat": 2
        },
        "qwen3-0.6b": {
            "chat": 2
        },
        "qwen-plus": {
            "chat": 2
        }, # 2025-04-28
        "qwen-turbo": {
            "chat": 2
        }, # 2025-04-28
        "Qwen2.5-Max": {
            "chat": 1
        },
        "qwen-max": {
            "chat": 1
        },
        "qwen2.5-72b-instruct": {
            "chat": 1
        },
        "qwen2.5-32b-instruct": {
            "chat": 1
        },
        "qwen2.5-14b-instruct": {
            "chat": 1
        },
        "qwen2.5-7b-instruct": {
            "chat": 1
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: QwenConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.enable_thinking = config.enable_thinking

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Default
                    completion = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "user", "content": prepared_text}],
                        )
                    summary = completion.choices[0].message.content
                case 2: # Reasoning model
                    completion = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_body = {"enable_thinking": self.enable_thinking},
                        messages=[
                            {"role": "user", "content": prepared_text}],
                        )
                    summary = completion.choices[0].message.content
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
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = OpenAI(
                    api_key=api_key, 
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
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