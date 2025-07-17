import os
from typing import Literal

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "microsoft"
class MicrosoftConfig(BasicLLMConfig):
    """Extended config for Microsoft-specific properties"""
    company: Literal["microsoft"] 
    model_name: Literal[
        ""
    ] # Only model names manually added to this list are supported.
    date_code: str = "" # do we need date code?
    execution_mode: Literal["api"] = "api" # Is this company only API based?
    endpoint: Literal["chat", "response"] = "chat"

class MicrosoftSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class MicrosoftLLM(AbstractLLM):
    """
    Class for models from Meta
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        ""
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: MicrosoftConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    pass
                case 2:
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
                assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = ChatCompletionsClient(
                    endpoint=endpoint,  # Of the form https://<your-host-name>.<your-azure-region>.models.ai.azure.com
                    credential=AzureKeyCredential(api_key)
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