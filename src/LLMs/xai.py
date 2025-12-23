import os
from typing import Literal
from xai_sdk import Client
from xai_sdk.chat import user, system
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "xai-org"
class XAIConfig(BasicLLMConfig):
    company: Literal["xai-org"] = "xai-org" 
    model_name: Literal[
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
        "grok-4",
        "grok-3",
        "grok-3-mini",
        "grok-3-fast",
        "grok-3-mini-fast",
        "grok-2-vision"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    reasoning_effort: Literal["NA", "low", "high"] = "NA"

class XAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

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
    "grok-4-1-fast-reasoning":{
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4-1-fast-non-reasoning":{
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4-fast-non-reasoning":{
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4-fast-reasoning":{
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4-fast-non-reasoning":{
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-4":{
        "chat": ClientMode.CHAT_REASONING
    },
    "grok-3":{
        "chat": ClientMode.CHAT_DEFAULT
    },
    "grok-3-mini":{
        "chat": ClientMode.CHAT_DEFAULT
    },
    "grok-3-fast":{
        "chat": ClientMode.CHAT_DEFAULT
    },
    "grok-3-mini-fast":{
        "chat": ClientMode.CHAT_DEFAULT
    },
    "grok-2-vision":{
        "chat": ClientMode.CHAT_DEFAULT
    },
}

local_mode_group = {}

class XAILLM(AbstractLLM):
    """
    Class for models from x-ai
    """
    def __init__(self, config: XAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.reasoning_effort = config.reasoning_effort
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_REASONING:
                    chat = self.client.chat.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    chat.append(user(prepared_text))

                    response = chat.sample()
                    summary = response.content
                    self.thinking_tokens = response.usage.reasoning_tokens
                case ClientMode.CHAT_DEFAULT:
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
            if self.model_name in client_mode_group:
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
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass