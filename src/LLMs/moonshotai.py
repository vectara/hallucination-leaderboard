
import os
from typing import Literal


from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from huggingface_hub import InferenceClient

"""
Unique Notes:
"""

COMPANY = "moonshotai"
class MoonshotAIConfig(BasicLLMConfig):
    """Extended config for moonshotai-specific properties"""
    company: Literal["moonshotai"] 
    model_name: Literal[
        "Kimi-K2-Instruct"
    ] # Only model names manually added to this list are supported.
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().

class MoonshotAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class MoonshotAILLM(AbstractLLM):
    """
    Class for models from moonshotai
    """
    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "Kimi-K2-Instruct": {
            "chat": 1
        },
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: MoonshotAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.huggingface_name = f"moonshotai/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Default
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text":  prepared_text}]}
                    ]
                    response = self.client.chat.completions.create(
                        model=self.huggingface_name,
                        messages=messages,
                        stream=False,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = response.choices[0].message.content
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
            if self.model_name in self.client_mode_group:
                self.client = InferenceClient(model=self.huggingface_name)
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
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass