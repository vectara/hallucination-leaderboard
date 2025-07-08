import os
from typing import Literal

from openai import OpenAI

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "openai"

class OpenAIConfig(BasicLLMConfig):
    """Extended config for OpenAI-specific properties"""
    company: Literal["openai"]
    model_name: Literal["gpt-4.1", "gpt-4.1-nano", "o3", "o3-pro"] # Only model names manually added to this list are supported.
    execution_mode: Literal["api"] = "api" # OpenAI models can only be run via web api.
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().

class OpenAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class OpenAIJudgment(BasicJudgment):
    pass # OpenAI does not have fields beyond BasicJudgment.

class OpenAILLM(AbstractLLM):
    """
    Class for models from OpenAI

    Attributes:
        client (OpenAI): client associated with api calls
        model_name (str): OpenAI style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    # Mode 1: Chat with temperature (default)
    # Mode 2: Chat without temperature
    # Mode 3: Use OpenAI's Response API
    client_mode_group = {
        "gpt-4.1": {
            "chat": 1,
            "response": 3
        },
        "gpt-4.1-nano": {
            "chat": 1,
            "response": 3
        },
        "o3": {  # o3 does not support temperature
            "chat": 2,
            "response": 3
        },
        "o3-pro": { # o3-pro doesn't support chatting protocol
            "chat": None, 
            "response": 3
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} # Empty for OpenAI models because they cannot be run locally.

    def __init__(self, config: OpenAIConfig):

        # Call parent constructor to inherit all parent properties
        super().__init__(config)

        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

        # Set default values for optional attributes
        # self.endpoint = config.endpoint if config.endpoint is not None else "chat" 
        # self.execution_mode = config.execution_mode if config.execution_mode is not None else "api"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Chat with temperature
                    chat_package = self.client.chat.completions.create(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content":prepared_text}]
                    )
                    summary = chat_package.choices[0].message.content
                case 2: # Chat without temperature
                    chat_package = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens
                    )
                    summary = chat_package.choices[0].message.content
                case 3: # Use OpenAI's Response API
                    chat_package = self.client.responses.create(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text
                    )
                    summary = chat_package.output_text
                case None:
                    raise Exception(f"Model `{self.model_name}` cannot be run from `{self.endpoint}` endpoint")
        elif self.local_model:
            pass # OpenAI models cannot be run locally.
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"OpenAI API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = OpenAI(api_key=api_key)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode == "local":
            pass # OpenAI models cannot be run locally.

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass # OpenAI models cannot be run locally.

    def close_client(self):
        pass
