import os
from typing import Literal

from mistralai import Mistral

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

"""
Notes:
Mistral Small 3.2: 2506
Mistral Small 3.1: 2503
Mistral Small 3: 2501
Mistral Small 2: 2407 | Doesnt work as of 7/16/2025. Reports invalid model
Mistral Large 2.1: 2411
Ministral 3B: 2410
Ministral 8b: 2410
Pixtral Large: 2411
Pixtral 12b: 2409
"""

COMPANY = "mistralai"

class MistralAIConfig(BasicLLMConfig):
    """Extended config for MistralAI-specific properties"""
    company: Literal["mistralai"] = "mistralai"
    model_name: Literal[
        "Ministral-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Mistral-Nemo-Instruct",
        "Mistral-Small-3.1-24b-instruct",
        "Mistral-Small-24B-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x22B-Instruct-v0.1",
        "Pixtral-Large-Instruct",
        


        "magistral-medium", 
        "mistral-medium",
        "mistral-small",
        "mistral-large",
        "ministral-3b",
        "ministral-8b",
        "pixtral-large",
        "pixtral-12b",
        "open-mistral-nemo"
    ] # Only model names manually added to this list are supported.
    execution_mode: Literal["api"] = "api" # MistralAI models can only be run via web api.
    date_code: str = "" # You must specify a date code for MistralAI models.
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().

class MistralAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore" 

class MistralAILLM(AbstractLLM):
    """
    Class for models from MistralAI

    Attributes:
        client (Mistral): client associated with api calls
        model_name (str): MistralAI style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {
        "magistral-medium":{
            "chat": 1
        }, # Doesn't look like magistral can disable thinking
        "mistral-medium":{
            "chat": 1
        },
        "mistral-small": {
            "chat": 1
        },
        "mistral-large": {
            "chat": 1
        },
        "ministral-3b": {
            "chat": 1
        },
        "ministral-8b": {
            "chat": 1
        },
        "pixtral-large": {
            "chat": 1
        },
        "pixtral-12b": {
            "chat": 1
        },
        "open-mistral-nemo": {
            "chat": 1
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} # Empty for MistralAI models because they cannot be run locally.

    def __init__(self, config: MistralAIConfig):
        # Ensure that the parameters passed into the constructor are of the type MistralAIConfig.
        
        # Call parent constructor to inherit all parent properties
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Standard chat completion
                    chat_package = self.client.chat.complete(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    summary = chat_package.choices[0].message.content
        elif self.local_model:
            pass # MistralAI models cannot be run locally.
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"MistralAI API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = Mistral(api_key=api_key)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode == "local":
            pass # MistralAI models cannot be run locally.

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass # MistralAI models cannot be run locally.

    def close_client(self):
        pass