import os
from typing import Literal
from enum import Enum, auto

import cohere

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "CohereLabs"
class CohereConfig(BasicLLMConfig):
    company: Literal["CohereLabs"] = "CohereLabs"
    model_name: Literal[
        "aya-expanse-8b",
        "aya-expanse-32b",
        "c4ai-command-r-plus",
        "command",
        "command-chat",
        "command-a",
        "command-a-reasoning",
        "c4ai-aya-expanse-32b",
        "c4ai-aya-expanse-8b",
        "command-r-plus",
        "command-r", 
        "command-r7b"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class CohereSummary(BasicSummary):
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
    "command-a": {
        "chat": 1
    },
    "command-a-reasoning": {
        "chat": 2
    },
    "c4ai-aya-expanse-32b": {
        "chat": 1
    },
    "c4ai-aya-expanse-8b": {
        "chat": 1
    },
    "command-r-plus": {
        "chat": 1
    },
    "command-r": {
        "chat": 1
    },
    "command-r7b": {
        "chat": 1
    },
}

local_mode_group = {}

class CohereLLM(AbstractLLM):
    """
    Class for models from cohere
    """
    def __init__(self, config: CohereConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    response = self.client.chat(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )

                    summary = response.message.content[0].text
                case 2:
                    response = self.client.chat(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    for content in response.message.content:
                        if content.type == "text":
                            summary = content.text
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
                api_key = os.getenv(f"COHERE_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = cohere.ClientV2(api_key=api_key)
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