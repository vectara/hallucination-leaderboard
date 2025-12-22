import os
from typing import Literal
from enum import Enum, auto

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "microsoft"
class MicrosoftConfig(BasicLLMConfig):
    company: Literal["microsoft"] 
    model_name: Literal[
        "Orca-2-13b",
        "phi-2",
        "Phi-3-mini-4k-instruct",
        "Phi-3-mini-128k-instruct",
        "Phi-3.5-mini-instruct",
        "Phi-3.5-MoE-instruct",
        "phi-4",
        "WizardLM-2-8x22B",


        "Phi-4-mini-instruct",
        "Phi-4",
        "microsoft-phi-2", # Resource not active
        "microsoft-Orca-2-13b" # Resource not active
    ]
    model_key: str = "NoneGiven"
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    azure_endpoint: str = "NoneGiven"
    endpoint: Literal["chat", "response"] = "chat"

class MicrosoftSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()
    # TODO: Add more as needed, make the term descriptive
class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()
    # TODO: Add more as needed, make the term descriptive

client_mode_group = {
    "Phi-4-mini-instruct": {
        "chat": 1
    },
    "Phi-4": {
        "chat": 1
    },
    "microsoft-phi-2": {
        "chat": 1
    },
    "microsoft-Orca-2-13b": {
        "chat": 1
    }
}

local_mode_group = {}

class MicrosoftLLM(AbstractLLM):
    """
    Class for models from Meta
    """
    def __init__(self, config: MicrosoftConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.azure_endpoint = config.azure_endpoint
        self.model_key = config.model_key

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    response = self.client.complete(
                        messages=[
                            UserMessage(content=prepared_text),
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        model=self.model_fullname
                    )

                    summary = response.choices[0].message.content
                case 2:
                    pass
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
                api_key = self.model_key
                assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = ChatCompletionsClient(
                    endpoint=self.azure_endpoint,
                    credential=AzureKeyCredential(api_key),
                    api_version="2024-05-01-preview"
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