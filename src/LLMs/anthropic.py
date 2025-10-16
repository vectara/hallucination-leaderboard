import os
from typing import Literal

import anthropic

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "anthropic"
class AnthropicConfig(BasicLLMConfig):
    """Extended config for Anthropic-specific properties"""
    company: Literal["anthropic"] = "anthropic"
    model_name: Literal[
        "claude-sonnet-4-5",
        "claude-opus-4-1", # 20250805
        "claude-haiku-4-5",
        "claude-3-5-haiku",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-7-sonnet", # 20250219 
        "claude-3-5-sonnet", # 20241022/20240620
        "claude-3-sonnet",
        "claude-3-opus",
        "claude-2.0"
    ] # Only model names manually added to this list are supported.
    date_code: str = "" # You must specify a date code for anthropic models.
    execution_mode: Literal["api"] = "api" # Anthropic models can only be run via web api. Actual default value set below in class `AnthropicLLM`.
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().
    class Config:
        extra = "forbid"

class AnthropicSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class AnthropicLLM(AbstractLLM):
    """
    Class for models from Anthropic
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
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

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} # Empty for Anthropic models because they cannot be run locally.

    def __init__(self, config: AnthropicConfig):
        
        # Call parent constructor to inherit all parent properties
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
            pass # Anthropic models cannot be run locally.
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        # elif self.local_model_is_defined():
        #     if False:
        #         pass
        #     else:
        #         raise LocalModelProtocolBranchNotFound(self.model_name)
        # else:
        #     raise ClientOrLocalNotInitializedError(self.model_name)
        return summary

    # def setup(self):
    #     if self.valid_client_model():
    #         api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
    #         self.client = anthropic.Client(api_key=api_key)
    #     elif self.valid_local_model():
    #         pass

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
            pass # Anthropic models cannot be run locally.

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass # Anthropic models cannot be run locally.

    def close_client(self):
        pass