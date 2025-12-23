import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
from enum import Enum, auto

# TODO: IMPORTS IF NEEDED

COMPANY = "COMPANY_NAME" # TODO: Using name from huggingface
class COMPANY_NAMEConfig(BasicLLMConfig): # TODO: Update object name
    company: Literal["COMPANY_NAME"] = "COMPANY_NAME" # TODO: Same name as COMPANY
    model_name: Literal[
        "MODEL_NAME", # TODO: Add models
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    # TODO: Add new attributes if needed for further customization

class COMPANY_NAMESummary(BasicSummary): # TODO: Update object name
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    CHAT_DEFAULT = auto() # LLM responds in a conversational manner
    RESPONSE_DEFAULT = auto() # LLM returns tokens until complete
    UNDEFINED = auto() # LLM has no defined case currently
    # TODO: Add more as needed, make the term descriptive.
    # Default refers to our default parameters being set (temperature, and tokens)

class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()
    # TODO: Add more as needed, make the term descriptive

# TODO: Add API models here to specify what logic path to run that model from
client_mode_group = {
    "MODEL_NAME": { 
        "chat": ClientMode.UNDEFINED
    }
}

class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    COMPLETION_DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

# TODO: Add local models here and specify what logic path to run that model
local_mode_group = {
    "MODEL_NAME": {
        "chat": LocalMode.UNDEFINED
    }
} 

class COMPANY_NAMELLM(AbstractLLM): # TODO: Update object name
    """
    """
    def __init__(self, config: COMPANY_NAMEConfig): # TODO: Update config name
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                # TODO Define how the default case model will run
                case ClientMode.CHAT_DEFAULT:
                    pass
        elif self.local_model: 
            match local_mode_group[self.model_name][self.endpoint]:
                # TODO Define how the default case model will run
                case LocalMode.CHAT_DEFAULT:
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
                # TODO: Assign client if using client based models
                self.client = None 
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )
        elif self.execution_mode == "local":
            if self.model_name in self.local_mode_group:
                # TODO: Assign a local model if using a local model
                self.local_model = None
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )

    def teardown(self):
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass