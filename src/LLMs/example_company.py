import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

# TODO: IMPORTS IF NEEDED

COMPANY = "COMPANY_NAME" # TODO: Provice EXACT company name
class COMPANY_NAMEConfig(BasicLLMConfig): # TODO: Edit name
    """Extended config for COMPANY_NAME-specific properties""" # TODO: Update comment
    company: Literal["COMPANY_NAME"] = "COMPANY_NAME" # TODO: Same name as COMPANY
    model_name: Literal[
        "MODEL_NAME", # TODO: Add models
    ] # Only model names manually added to this list are supported.
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api" # GPU refers to a local model
    endpoint: Literal["chat", "response"] = "chat"
    # TODO: Add new attributes if needed for further customization

class COMPANY_NAMESummary(BasicSummary): # TODO: Update object name
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore"

class COMPANY_NAMELLM(AbstractLLM): # TODO: Update object name
    """
    Class for models from COMPANY_NAME
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "MODEL_NAME": { # TODO: Add API models here to specify what logic path to run that model from
            "chat": 1
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} #TODO: Add local models here and specify what logic path to run that model

    def __init__(self, config: COMPANY_NAMEConfig): # TODO: Update config
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        # Use self.model_fullname when referring to the model
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                #TODO Define how the case 1 model will run
                case 1:
                    summary = None 
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
                self.client = None # TODO: Assign client if using client based models
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )
        elif self.execution_mode == "local":
            # TODO: Assign a local model if using a local model
            self.local_model = None
            pass

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass