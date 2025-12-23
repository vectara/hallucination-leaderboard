import os
import torch
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
import re
import gc
import replicate

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPANY = "ibm-granite"

class IBMGraniteConfig(BasicLLMConfig):
    company: Literal["ibm-granite"] = "ibm-granite"
    model_name: Literal[
        "granite-4.0-h-small",
        "granite-4.0-h-tiny",
        "granite-4.0-h-micro",
        "granite-4.0-micro",
        "granite-3.3-8b-instruct",
        "granite-3.2-8b-instruct",
        "granite-3.2-2b-instruct",
        "granite-3.1-8b-instruct",
        "granite-3.1-2b-instruct",
        "granite-3.0-8b-instruct",
        "granite-3.0-2b-instruct"
    ]
    endpoint: Literal["chat", "response"] = "chat"
    execution_mode: Literal["api", "gpu", "cpu"] = "api"

class IBMGraniteSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    CHAT_DEFAULT = auto()
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()
    # TODO: Add more as needed, make the term descriptive
class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()
    # TODO: Add more as needed, make the term descriptive

client_mode_group = {
    "granite-4.0-h-small": {
        "chat": ClientMode.CHAT_DEFAULT,
    },
    "granite-3.3-8b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "granite-3.2-8b-instruct": {
        "chat": ClientMode.CHAT_DEFAULT
    },

}

local_mode_group = {
    "granite-4.0-h-small": {
        "chat": ClientMode.UNDEFINED,
    },
    "granite-4.0-h-tiny": {
        "chat": ClientMode.UNDEFINED,
    },
    "granite-4.0-h-micro": {
        "chat": ClientMode.UNDEFINED,
    },
    "granite-4.0-micro": {
        "chat": ClientMode.UNDEFINED,
    },
    "granite-3.2-8b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.2-2b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.1-8b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.1-2b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.0-8b-instruct": {
        "chat": ClientMode.UNDEFINED
    },
    "granite-3.0-2b-instruct": {
        "chat": ClientMode.UNDEFINED
    }
}

class IBMGraniteLLM(AbstractLLM):
    """
    Class for models from IBM
    """
    def __init__(self, config: IBMGraniteConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_fullname = f"{COMPANY}/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT: # Default
                    input = {
                        "prompt": prepared_text,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                    }
                    raw_out = replicate.run(
                        f"{self.model_fullname}",
                        input=input
                    )
                    summary = raw_out[0]
        elif self.local_model:
            match local_mode_group[self.model_name][self.endpoint]:
                case 1: # Uses chat template
                    pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            self.client = "replicate doesnt have a client"
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
                pass
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))

    def teardown(self):
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass