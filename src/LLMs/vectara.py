import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "vectara"
class VectaraConfig(BasicLLMConfig):
    """Extended config for vectara-specific properties"""
    company: Literal["vectara"] = "vectara" 
    model_name: Literal[
        "manual_short_summary",
        "manual_long_summary",
    ] # Only model names manually added to this list are supported.
    date_code: str = "" # do we need date code?
    execution_mode: Literal["api"] = "api" # only API based?
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().

class VectaraSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class VectaraLLM(AbstractLLM):
    """
    Class for models from vectara
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "manual_short_summary":{
            "chat": 0
        },
        "manual_long_summary":{
            "chat": 0
        },
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: VectaraConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.reasoning_effort = config.reasoning_effort
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 0:
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
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = None
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