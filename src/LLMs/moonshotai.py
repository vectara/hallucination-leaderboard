
import os
from typing import Literal
from enum import Enum, auto


from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from huggingface_hub import InferenceClient
from together import Together
from openai import OpenAI

COMPANY = "moonshotai"
class MoonshotAIConfig(BasicLLMConfig):
    company: Literal["moonshotai"] 
    model_name: Literal[
        "Kimi-K2-Instruct",
        "kimi-k2-thinking"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class MoonshotAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
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
    "Kimi-K2-Instruct": {
        "chat": 2
    },
    "kimi-k2-thinking": {
        "chat": 1
    },
}

local_mode_group = {}

class MoonshotAILLM(AbstractLLM):
    """
    Class for models from moonshotai
    """
    def __init__(self, config: MoonshotAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.huggingface_name = f"moonshotai/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case 1: # Default
                    completion = self.client.chat.completions.create(
                        model = self.model_fullname,
                        messages = [
                            {"role": "user", "content": prepared_text}
                        ],
                        temperature = self.temperature,
                        max_tokens = self.max_tokens
                    )
                    
                    summary = completion.choices[0].message.content

                case 2:
                    if self.date_code == "0905":
                        client = Together()
                        response = client.chat.completions.create(
                        model=self.huggingface_name,
                            messages=[
                                {
                                "role": "user",
                                "content": prepared_text
                                }
                            ],
                            max_tokens = self.max_tokens,
                            temperature = self.temperature,
                        )
                        summary = response.choices[0].message.content
                    else:
                        messages = [
                            {"role": "user", "content": [{"type": "text", "text":  prepared_text}]}
                        ]
                        response = self.client.chat.completions.create(
                            model=self.huggingface_name,
                            messages=messages,
                            stream=False,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        summary = response.choices[0].message.content
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
            api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
            assert api_key is not None, f"{COMPANY.upper()} API key not found in environment variable {COMPANY.upper()}_API_KEY"
            if self.model_name in client_mode_group:
                self.client = OpenAI(
                    api_key = api_key,
                    base_url = "https://api.moonshot.ai/v1",
                )
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