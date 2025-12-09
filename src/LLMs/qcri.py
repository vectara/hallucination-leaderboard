import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from openai import OpenAI

COMPANY = "qcri"

class QCRIConfig(BasicLLMConfig):
    """Extended config for Fanar-specific properties"""
    company: Literal["qcri"] = "qcri"
    model_name: Literal["fanar-model"]
    execution_mode: Literal["api"] = "api"
    date_code: str

class QCRISummary(BasicSummary):
    pass

class QCRIJudgment(BasicJudgment):
    pass

class QCRILLM(AbstractLLM):
    """
    Class for models from Fanar
    """

    client_mode_group = {
        "Fanar": 1
    }

    local_mode_group = {}

    def __init__(self, config: QCRIConfig):
        super().__init__(config)

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name]:
                case 1: # Standard chat completion
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content":prepared_text}]
                    )
                    summary = chat_package.choices[0].message.content
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"Fanar API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = OpenAI(
                    base_url="https://api.fanar.qa/v1",
                    api_key=api_key
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
