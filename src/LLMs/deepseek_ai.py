import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from huggingface_hub import InferenceClient

COMPANY = "deepseek-ai"

class DeepSeekAIConfig(BasicLLMConfig):
    company: Literal["deepseek-ai"] = "deepseek-ai"
    model_name: Literal[
        "DeepSeek-V3.1-Terminus",
        "DeepSeek-V3.2",
        "DeepSeek-V3.2-Exp",
        "deepseek-chat",
        "deepseek-coder",
        "DeepSeek-R1-0528",
        "DeepSeek-V3",
        "DeepSeek-V3.1",
        "DeepSeek-R1",
        "DeepSeek-V2.5" # Not implemented


    ]
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"

class DeepSeekAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    CHAT_DEFAULT = auto()
    CHAT_NO_TEMP_NO_TOKENS = auto()
    CHAT_CONVERSATIONAL_NO_TOKENS = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

client_mode_group = {
    "DeepSeek-V3.2": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-R1": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V3": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V3.1": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V3.1-Terminus": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V3.2-Exp": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "DeepSeek-V2.5": {
        "chat": ClientMode.CHAT_NO_TEMP_NO_TOKENS
    }
}

local_mode_group = {}


class DeepSeekAILLM(AbstractLLM):
    """
    Class for models from DeepSeekAI
        model_name (str): DeepSeekAI style model name
    """
    def __init__(self, config: DeepSeekAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.model_fullname = f"{self.company}/{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
                    messages = [{"role": "user", "content":prepared_text}]
                    client_package = self.client.chat_completion(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = client_package.choices[0].message.content
                case ClientMode.CHAT_NO_TEMP_NO_TOKENS:
                    messages = [{"role": "user", "content":prepared_text}]
                    client_package = self.client.chat_completion(
                        messages
                    )
                    summary = client_package.choices[0].message.content
                case ClientMode.CHAT_CONVERSATIONAL_NO_TOKENS:
                    client_package = self.client.conversational(
                        messages=prepared_text,
                        temperature=self.temperature
                    )
                    summary=client_package["generated_text"]
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                self.client = InferenceClient(model=self.model_fullname)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode == "local":
            pass

    def teardown(self):
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass
