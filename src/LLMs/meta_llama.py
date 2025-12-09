import os
from typing import Literal

from together import Together

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "meta-llama"
class MetaLlamaConfig(BasicLLMConfig):
    company: Literal["meta-llama"] = "meta-llama" 
    model_name: Literal[
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-70b-chat-hf",
        "Llama-3-8B-chat-hf",
        "Llama-3-70B-chat-hf",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct-Turbo",
        "Llama-3.2-11B-Vision-Instruct-Turbo",
        "Llama-3.2-90B-Vision-Instruct-Turbo",
        "Llama-3.3-70B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",

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
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class MetaLlamaSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class MetaLlamaLLM(AbstractLLM):
    """
    Class for models from Meta
    """

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

    local_mode_group = {}

    def __init__(self, config: MetaLlamaConfig):
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
                api_key = os.getenv(f"TOGETHER_API_KEY")
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
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass