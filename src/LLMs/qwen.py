import os
from typing import Literal
from enum import Enum, auto

from openai import OpenAI

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "qwen"
class QwenConfig(BasicLLMConfig):
    company: Literal["qwen"] 
    model_name: Literal[
        "Qwen2-72B-Instruct",
        "Qwen2-VL-2B-Instruct",
        "Qwen2-VL-7B-Instruct",
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-30B-A3B",
        "Qwen3-32B",
        "Qwen3-235B-A22B",
        "QwQ-32B-Preview",

        "qwen3-30b-a3b-thinking",
        "qwen3-next-80b-a3b-thinking",
        "qwen3-omni-30b-a3b-thinking",

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
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    thinking_tokens: bool = None
    enable_thinking: bool = None

class QwenSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None
    enable_thinking: bool | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    CHAT_DEFAULT = auto()
    CHAT_REASONING = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

client_mode_group = {
    "Qwen3-235B-A22B": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-30b-a3b-thinking": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-next-80b-a3b-thinking": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-omni-30b-a3b-thinking": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-max-preview": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-32b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-14b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-8b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-4b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-1.7b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen3-0.6b": {
        "chat": ClientMode.CHAT_REASONING
    },
    "qwen-plus": {
        "chat": ClientMode.CHAT_REASONING
    }, # 2025-04-28
    "qwen-turbo": {
        "chat": ClientMode.CHAT_REASONING
    }, # 2025-04-28
    "Qwen2.5-Max": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen-max": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen2.5-72b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen2.5-32b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen2.5-14b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "qwen2.5-7b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}

local_mode_group = {}

class QwenLLM(AbstractLLM):
    def __init__(self, config: QwenConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.enable_thinking = config.enable_thinking

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
                    completion = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "user", "content": prepared_text}],
                        )
                    summary = completion.choices[0].message.content
                case ClientMode.CHAT_REASONING:
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
            if self.model_name in client_mode_group:
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
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass