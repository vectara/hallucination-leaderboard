import os
from typing import Literal
from enum import Enum, auto

from mistralai import Mistral

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "mistralai"

class MistralAIConfig(BasicLLMConfig):
    company: Literal["mistralai"] = "mistralai"
    model_name: Literal[
        "Ministral-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Mistral-Nemo-Instruct",
        "Mistral-Small-3.1-24b-instruct",
        "Mistral-Small-24B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x22B-Instruct-v0.1",
        "Pixtral-Large-Instruct",
        


        "magistral-medium", 
        "mistral-medium",
        "mistral-small",
        "mistral-large",
        "ministral-3b",
        "ministral-8b",
        "ministral-14b",
        "pixtral-large",
        "pixtral-12b",
        "open-mistral-nemo"
    ]
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"

class MistralAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore" 

class ClientMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

client_mode_group = {
    "magistral-medium":{
        "chat": ClientMode.CHAT_DEFAULT
    },
    "mistral-medium":{
        "chat": ClientMode.CHAT_DEFAULT
    },
    "mistral-small": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "mistral-large": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "ministral-3b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "ministral-8b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "ministral-14b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "pixtral-large": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "pixtral-12b": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "open-mistral-nemo": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}

local_mode_group = {}

class MistralAILLM(AbstractLLM):
    """
    Class for models from MistralAI
    """
    def __init__(self, config: MistralAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
                    chat_package = self.client.chat.complete(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    summary = chat_package.choices[0].message.content
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"MistralAI API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = Mistral(api_key=api_key)
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