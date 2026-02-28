import os
import requests
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "inceptionlabs"


class InceptionLabsConfig(BasicLLMConfig):
    company: Literal["inceptionlabs"] = "inceptionlabs"
    model_name: Literal[
        "mercury-2",
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    api_type: Literal["default"] = "default"


class InceptionLabsSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None
    api_type: Literal["default"] | None = None

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
    "mercury-2": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}

local_mode_group = {}


class InceptionLabsLLM(AbstractLLM):
    def __init__(self, config: InceptionLabsConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
                    response = requests.post(
                        'https://api.inceptionlabs.ai/v1/chat/completions',
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {self.client}'
                        },
                        json={
                            'model': self.model_fullname,
                            'messages': [
                                {'role': 'user', 'content': prepared_text}
                            ],
                            'max_tokens': self.max_tokens,
                            'temperature': self.temperature,
                        }
                    )
                    data = response.json()
                    summary = data['choices'][0]['message']['content']
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
                if self.api_type == "default":
                    api_key = os.getenv("INCEPTIONLABS_API_KEY")
                    assert api_key is not None, (
                        "InceptionLabs API key not found in environment variable "
                        "INCEPTIONLABS_API_KEY"
                    )
                    self.client = api_key
                else:
                    raise ValueError(f"Unknown api_type: {self.api_type}")
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
