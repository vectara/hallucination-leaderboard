import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

# Import the Python package for the specific provider.
from huggingface_hub import InferenceClient

COMPANY = "deepseek-ai"

class DeepSeekAIConfig(BasicLLMConfig):
    """Extended config for DeepSeekAI-specific properties"""
    company: Literal["deepseek-ai"] = "deepseek-ai"
    model_name: Literal[
        "DeepSeek-V3.2-Exp",
        "deepseek-chat",
        "deepseek-coder",
        "DeepSeek-R1-0528",
        "DeepSeek-V3", #0324
        "DeepSeek-V3.1",
        "DeepSeek-R1",
        "DeepSeek-V2.5" # Not implemented


    ] # Only model names manually added to this list are supported.
    execution_mode: Literal["api"] = "api" # DeepSeekAI models can only be run via web api.
    date_code: str = "" # You must specify a date code for DeepSeekAI models.
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().

class DeepSeekAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class DeepSeekAILLM(AbstractLLM):
    """
    Class for models from DeepSeekAI

    Attributes:
        client (InferenceClient): client associated with api calls
        model_name (str): DeepSeekAI style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {
        "DeepSeek-R1": {
            "chat": 1
        },
        "DeepSeek-V3": {
            "chat": 1
        },
        "DeepSeek-V3.1": {
            "chat": 1
        },
        "DeepSeek-V3.2-Exp": {
            "chat": 1
        },
        "DeepSeek-V2.5": {
            "chat": 2
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} # Empty for DeepSeekAI models because they cannot be run locally.

    def __init__(self, config: DeepSeekAIConfig):
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
                case 2: # V2.5. Does not work
                    messages = [{"role": "user", "content":prepared_text}]
                    client_package = self.client.chat_completion(
                        messages
                    )
                    summary = client_package.choices[0].message.content
        elif self.local_model:
            pass # DeepSeekAI models cannot be run locally.
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
            pass # DeepSeekAI models cannot be run locally.

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass # DeepSeekAI models cannot be run locally.

    def close_client(self):
        pass
