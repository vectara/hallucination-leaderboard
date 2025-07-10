import os
from typing import Literal
from pydantic import field_validator

from openai import OpenAI

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "alibaba"
class AlibabaConfig(BasicLLMConfig):
    """Extended config for Alibaba-specific properties"""
    company: Literal["alibaba"] 
    model_name: Literal[
        "qwen3-32b",
        "qwen3-14b",
        "qwen3-8b",
        "qwen3-4b",
        "qwen3-1.7b",
        "qwen3-0.6b"
    ] # Only model names manually added to this list are supported.
    date_code: str = "" # do we need date code for ali baba?
    execution_mode: Literal["api"] = "api" # Is Alibaba only API based?
    thinking_tokens: bool

class AlibabaSummary(BasicSummary):
    pass # Nothing additional to the BasicSummary class.

class AlibabaLLM(AbstractLLM):
    """
    Class for models from Alibaba
    """
    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "qwen3-32b": 1,
        "qwen3-14b": 1,
        "qwen3-8b": 1,
        "qwen3-4b": 1,
        "qwen3-1.7b": 1,
        "qwen3-0.6b": 1
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: AlibabaConfig):
        super().__init__(config)

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name]:
                case 1: # Reasoning model with disabled thinking
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_body = {"enable_thinking": self.thinking_tokens},
                        messages=[
                            {"role": "user", "content": prepared_text}],
                        )
                    summary = completion.choices[0].message.content
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
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = OpenAI(
                    api_key=api_key, 
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
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
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass