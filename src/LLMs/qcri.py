import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

#TODO: Rename  qcri

# Import the Python package for the specific provider.
from openai import OpenAI

COMPANY = "fanar"

class FanarConfig(BasicLLMConfig):
    """Extended config for Fanar-specific properties"""
    company: Literal["fanar"] = "fanar"
    model_name: Literal["fanar-model"] # Only model names manually added to this list are supported.
    execution_mode: Literal["api"] = "api" # Fanar models can only be run via web api.
    date_code: str # You must specify a date code for Fanar models.

class FanarSummary(BasicSummary):
    pass # Nothing additional to the BasicSummary class.

class FanarJudgment(BasicJudgment):
    pass # Fanar does not have fields beyond BasicJudgment.

class FanarLLM(AbstractLLM):
    """
    Class for models from Fanar

    Attributes:
        client (OpenAI): client associated with api calls
        model_name (str): Fanar style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {
        "Fanar": 1
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} # Empty for Fanar models because they cannot be run locally.

    def __init__(self, config: FanarConfig):
        # Ensure that the parameters passed into the constructor are of the type FanarConfig.
        
        # Call parent constructor to inherit all parent properties
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
            pass # Fanar models cannot be run locally.
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
            pass # Fanar models cannot be run locally.

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass # Fanar models cannot be run locally.

    def close_client(self):
        pass
