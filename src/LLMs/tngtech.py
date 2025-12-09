import os
from typing import Literal

from openai import OpenAI

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "tngtech"

class TngTechConfig(BasicLLMConfig):
    company: Literal["tngtech"]
    model_name: Literal[
        "DeepSeek-TNG-R1T2-Chimera"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class TngTechSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class TngTechLLM(AbstractLLM):
    """
    Class for models from TngTech
    """

    client_mode_group = {
        "DeepSeek-TNG-R1T2-Chimera": {
            "chat": 1,
        }
    }

    local_mode_group = {}

    def __init__(self, config: TngTechConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.model_fullname = f"{COMPANY}/{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content":prepared_text}]
                    )
                    summary = chat_package.choices[0].message.content
                case None:
                    raise Exception(f"Model `{self.model_name}` cannot be run from `{self.endpoint}` endpoint")
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://chat.model.tngtech.com/v1/"
                )
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