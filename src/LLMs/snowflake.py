import os
from typing import Literal
import replicate

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "snowflake" #Official company name on huggingface
class SnowflakeConfig(BasicLLMConfig):
    """Extended config for snowflake-specific properties"""
    company: Literal["snowflake"] = "snowflake"
    model_name: Literal[
        "snowflake-arctic-instruct",
    ] # Only model names manually added to this list are supported.
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class SnowflakeSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore"

class SnowflakeLLM(AbstractLLM):
    """
    Class for models from snowflake
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "snowflake-arctic-instruct": {
            "chat": 1
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: SnowflakeConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config
        self.model_fullname = f"{COMPANY}/{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        # Use self.model_fullname when referring to the model
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    input = {
                        "prompt": prepared_text,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                    }
                    raw_out = replicate.run(
                        f"{self.model_fullname}",
                        input=input
                    )
                    summary=raw_out[0]
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
            self.client = "Replicate doesn't have a client"
            # if self.model_name in self.client_mode_group:
            #     api_key = os.getenv(f"REPLICATE_API_TOKEN")
            #     assert api_key is not None, (
            #         f"{COMPANY} API key not found in environment variable "
            #         f"{COMPANY.upper()}_API_KEY"
            #     )
            #     self.client = "Replicate doesn't have a client"
            # else:
            #     raise Exception(
            #         ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
            #             model_name=self.model_name,
            #             company=self.company,
            #             execution_mode=self.execution_mode
            #         )
            #     )
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