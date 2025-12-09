import os
from typing import Literal

import anthropic

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "anthropic"
class AnthropicConfig(BasicLLMConfig):
    company: Literal["anthropic"] = "anthropic"
    model_name: Literal[
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-1",
        "claude-haiku-4-5",
        "claude-3-5-haiku",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "claude-3-sonnet",
        "claude-3-opus",
        "claude-2.0"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat" 
    class Config:
        extra = "forbid"

class AnthropicSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class AnthropicLLM(AbstractLLM):
    """
    Class for models from Anthropic
    """

    client_mode_group = {
        "claude-opus-4-5": {
            "chat": 1
        },
        "claude-3-5-haiku": {
            "chat": 1
        },
        "claude-sonnet-4-5": {
            "chat": 1
        },
        "claude-haiku-4-5": {
            "chat": 1
        },
        "claude-opus-4-1": {
            "chat": 1
        },
        "claude-opus-4": {
            "chat": 1
        },
        "claude-sonnet-4": {
            "chat": 1
        },
        "claude-3-7-sonnet": {
            "chat": 1
        },
        "claude-3-5-sonnet": {
            "chat": 1
        },
        "claude-3-sonnet": {
            "chat": 1
        },
        "claude-3-opus": {
            "chat": 1
        },
        "claude-2.0": {
            "chat": 1
        }
    }

    local_mode_group = {}

    def __init__(self, config: AnthropicConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    chat_package = self.client.messages.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    summary = chat_package.content[0].text
        elif self.local_model: 
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"Anthropic API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = anthropic.Anthropic(api_key=api_key)
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