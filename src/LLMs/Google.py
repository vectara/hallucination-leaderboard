import os
from typing import Literal

from google import genai
from google.genai import types

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "google"

class GoogleConfig(BasicLLMConfig):
    """Extended config for Google-specific properties"""
    company: Literal["google"] = "google"
    model_name: Literal["gemini-2.5-pro-preview", "gemini-2.5-pro"] # Only model names manually added to this list are supported.
    execution_mode: Literal["api"] = "api" # Google models can only be run via web api.
    date_code: str # You must specify a date code for Google models.

class GoogleSummary(BasicSummary):
    pass # Nothing additional to the BasicSummary class.

class GoogleJudgment(BasicJudgment):
    pass # Google does not have fields beyond BasicJudgment.

class GoogleLLM(AbstractLLM):
    """
    Class for models from Google

    Attributes:
        client (genai.Client): client associated with api calls
        model_name (str): Google style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {
        "gemini-2.5-pro-preview": 1, # Requires large output token amount, set to 4096
        "gemini-2.5-pro": 2 # Supports thinking tokens
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} # Empty for Google models because they cannot be run locally.

    def __init__(self, config: GoogleConfig):
        # Ensure that the parameters passed into the constructor are of the type GoogleConfig.
        
        # Call parent constructor to inherit all parent properties
        super().__init__(config)

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name]:
                case 1: # gemini-2.5-pro-preview - requires large output token amount
                    response = self.client.models.generate_content(
                        model=self.model_fullname,
                        contents=prepared_text,
                        config=types.GenerateContentConfig(
                            max_output_tokens=4096,
                            temperature=self.temperature
                        )
                    )
                    summary = response.text
                case 2: # gemini-2.5-pro - supports thinking tokens
                    response = self.client.models.generate_content(
                        model=self.model_fullname,
                        contents=prepared_text,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_tokens)
                        ),
                    )
                    summary = response.text
        elif self.local_model:
            pass # Google models cannot be run locally.
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_GEMINI_API_KEY")
                assert api_key is not None, f"Google Gemini API key not found in environment variable {COMPANY.upper()}_GEMINI_API_KEY"
                self.client = genai.Client(api_key=api_key)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode == "local":
            pass # Google models cannot be run locally.

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass # Google models cannot be run locally.

    def close_client(self):
        pass
