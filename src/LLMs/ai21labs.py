import os
from typing import Literal
import requests

from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "ai21labs"
class AI21LabsConfig(BasicLLMConfig):
    company: Literal["ai21labs"] = "ai21labs"
    model_name: Literal[
        "AI21-Jamba-Mini-1.5",
        "jamba-mini-2",
        "jamba-large-1.7",
        "jamba-mini-1.7",
        "jamba-large-1.6", # Deprecated
        "jamba-mini-1.6", # Deprecated
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    class Config:
        extra = "forbid"

class AI21LabsSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    CHAT_DEFAULT = auto()
    CHAT_HTTP = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

client_mode_group = {
    "jamba-mini-2": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "jamba-large-1.7": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "jamba-mini-1.7": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "jamba-large-1.6": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "jamba-mini-1.6": {
        "chat": ClientMode.CHAT_DEFAULT
    },
}

# In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
local_mode_group = {}

class AI21LabsLLM(AbstractLLM):
    """
    Class for models from AI21
    """
    def __init__(self, config: AI21LabsConfig):
        
        # Call parent constructor to inherit all parent properties
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
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
                case ClientMode.CHAT_HTTP:
                    api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                    url = "https://api.ai21.com/studio/v1/chat/completions"

                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }

                    payload = {
                        "model": self.model_fullname,
                        "messages": [
                            {"role": "user", "content": prepared_text}
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }

                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()

                    data = response.json()
                    summary = data["choices"][0]["message"]["content"]

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

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
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
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass