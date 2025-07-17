import os
from typing import Literal

from together import Together

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "meta"
class MetaConfig(BasicLLMConfig):
    """Extended config for Meta-specific properties"""
    company: Literal["meta"] 
    model_name: Literal[
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Llama-4-Scout-17B-16E-Instruct",
        "Meta-Llama-3.1-8B-Instruct-Turbo",
        "Llama-3.3-70B-Instruct-Turbo",
        "Llama-3.3-70B-Instruct-Turbo-Free",
        "Meta-Llama-3.1-405B-Instruct-Turbo",
        "Llama-3.2-3B-Instruct-Turbo",
        "Llama-3.2-11B-Vision-Instruct-Turbo*",
        "Llama-3.2-90B-Vision-Instruct-Turbo*",
        "Meta-Llama-3.1-405B-Instruct-Turbo",
        "Meta-Llama-3.1-8B-Instruct-Turbo",
        "Meta-Llama-3-8B-Instruct-Lite",
        "Llama-3-8b-chat-hf*",
        "Llama-3-70b-chat-hf",
        "Llama-2-70b-hf" # Completion?
    ] # Only model names manually added to this list are supported.
    date_code: str = "" # do we need date code?
    execution_mode: Literal["api"] = "api" # Is this company only API based?
    endpoint: Literal["chat", "response"] = "chat"

class MetaSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class MetaLLM(AbstractLLM):
    """
    Class for models from Meta
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "Llama-4-Maverick-17B-128E-Instruct-FP8": {
            "chat": 1
        },
        "Llama-4-Scout-17B-16E-Instruct": {
            "chat": 1
        },
        "Meta-Llama-3.1-8B-Instruct-Turbo": {
            "chat": 1
        },
        "Llama-3.3-70B-Instruct-Turbo": {
            "chat": 1
        },
        "Llama-3.3-70B-Instruct-Turbo-Free": {
            "chat": 1
        },
        "Meta-Llama-3.1-405B-Instruct-Turbo": {
            "chat": 1
        },
        "Llama-3.2-3B-Instruct-Turbo": {
            "chat": 1
        },
        "Llama-3.2-11B-Vision-Instruct-Turbo*": { # Unable to access model atm
            "chat": 0
        },
        "Llama-3.2-90B-Vision-Instruct-Turbo*": { # Unable to access model atm
            "chat": 0
        },
        "Meta-Llama-3.1-405B-Instruct-Turbo": {
            "chat": 1
        },
        "Meta-Llama-3.1-8B-Instruct-Turbo": {
            "chat": 1
        },
        "Meta-Llama-3-8B-Instruct-Lite": {
            "chat": 1
        },
        "Llama-3-8b-chat-hf*": {
            "chat": 1
        },
        "Llama-3-70b-chat-hf": {
            "chat": 1
        },
        "Llama-2-70b-hf": {
            "response": 2
        } # Completion?
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: MetaConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Default chat
                    together_name = f"meta-llama/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                    model=together_name,
                    messages=[{"role": "user", "content": prepared_text}],
                    max_tokens = self.max_tokens,
                    temperature = self.temperature
                    )
                    summary = response.choices[0].message.content

                case 2: # Default Completion
                    together_name = f"meta-llama/{self.model_fullname}"
                    response = self. client.completions.create(
                        model=together_name,
                        prompt=prepared_text,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    summary = response.choices[0].text
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
                api_key = os.getenv(f"{COMPANY.upper()}_TOGETHER_API_KEY")
                assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = Together(api_key=api_key)
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