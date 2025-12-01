import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

# Import the Python package for the specific provider.
from huggingface_hub import InferenceClient

COMPANY = "PrimeIntellect"

class PrimeIntellectConfig(BasicLLMConfig):
    """Extended config for company specific properties"""
    company: Literal["PrimeIntellect"] = "PrimeIntellect"
    model_name: Literal[
        "INTELLECT-3"
    ] # Only model names manually added to this list are supported.
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().

class PrimeIntellectSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore"

class PrimeIntellectLLM(AbstractLLM):
    """
    Class for models from company

    Attributes:
        client (InferenceClient): client associated with api calls
        model_name (str): company style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {
        "INTELLECT-3": {
            "chat": 1
        },
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} # Empty for DeepSeekAI models because they cannot be run locally.

    def __init__(self, config: PrimeIntellectConfig):
        # Ensure that the parameters passed into the constructor are of the type DeepSeekAIConfig.
        
        # Call parent constructor to inherit all parent properties
        super().__init__(config)
        # Construct the full model name with company prefix
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.model_fullname = f"{self.company}/{self.model_name}" # double check

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
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
            if self.model_name in self.client_mode_group:
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
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass
