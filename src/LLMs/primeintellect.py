import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from huggingface_hub import InferenceClient

COMPANY = "PrimeIntellect"

class PrimeIntellectConfig(BasicLLMConfig):
    company: Literal["PrimeIntellect"] = "PrimeIntellect"
    model_name: Literal[
        "INTELLECT-3"
    ]
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"

class PrimeIntellectSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

class LocalMode(Enum):
    DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

client_mode_group = {
    "INTELLECT-3": {
        "chat": 1
    },
}

local_mode_group = {}

class PrimeIntellectLLM(AbstractLLM):
    """
    Class for models from company
    """
    def __init__(self, config: PrimeIntellectConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.model_fullname = f"{self.company}/{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case 1: # Standard chat completion
                    messages = [{"role": "user", "content":prepared_text}]
                    client_package = self.client.chat_completion(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = client_package.choices[0].message.content
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
