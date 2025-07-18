import os
from typing import Literal

from ai21 import AI21Client
from ai21.models.chat import ChatMessage

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "ai21"
class AI21Config(BasicLLMConfig):
    """Extended config for AI21-specific properties"""
    company: Literal["ai21"] 
    model_name: Literal[
        "jamba-large-1.7",
        "jamba-mini-1.7",
        "jamba-large-1.6", # Deprecated
        "jamba-mini-1.6", # Deprecated
    ] # Only model names manually added to this list are supported.
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    class Config:
        extra = "forbid"

class AI21Summary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class AI21LLM(AbstractLLM):
    """
    Class for models from AI21
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "jamba-large-1.7": {
            "chat": 1
        },
        "jamba-mini-1.7": {
            "chat": 1
        },
        "jamba-large-1.6": {
            "chat": 1
        },
        "jamba-mini-1.6": {
            "chat": 1
        },
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: AI21Config):
        
        # Call parent constructor to inherit all parent properties
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    messages = [
                        ChatMessage(role="user", content=prepared_text),
                    ]
                    response = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_fullname,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )

                    summary = response.choices[0].message.content
        elif self.local_model: 
            pass 
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        # elif self.local_model_is_defined():
        #     if False:
        #         pass
        #     else:
        #         raise LocalModelProtocolBranchNotFound(self.model_name)
        # else:
        #     raise ClientOrLocalNotInitializedError(self.model_name)
        return summary

    # def setup(self):
    #     if self.valid_client_model():
    #         api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
    #         self.client = anthropic.Client(api_key=api_key)
    #     elif self.valid_local_model():
    #         pass

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = AI21Client(api_key=api_key)
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
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass