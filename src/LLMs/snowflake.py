import os
from typing import Literal
import replicate

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "snowflake"
class SnowflakeConfig(BasicLLMConfig):
    company: Literal["snowflake"] = "snowflake"
    model_name: Literal[
        "snowflake-arctic-instruct",
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class SnowflakeSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class SnowflakeLLM(AbstractLLM):
    """
    Class for models from snowflake
    """

    client_mode_group = {
        "snowflake-arctic-instruct": {
            "chat": 1
        }
    }

    local_mode_group = {}

    def __init__(self, config: SnowflakeConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config
        self.model_fullname = f"{COMPANY}/{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
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
        elif self.execution_mode == "local":
            pass

    def teardown(self):
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass