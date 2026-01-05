import os
from typing import Literal
from together import Together
from openai import OpenAI
from enum import Enum, auto


from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "zai-org"
class ZhipuAIConfig(BasicLLMConfig):
    company: Literal["zai-org"] = "zai-org" 
    model_name: Literal[
        "GLM-4.5-AIR-FP8", # Together
        "glm-4p5", # Fireworks but using OpenAI
        "glm-4p7", # Fireworks but using OpenAI
        "glm-4-9b-chat",
        "GLM-4.6",
        "GLM-4.7"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class ZhipuAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    CHAT_DEFAULT = auto()
    GLM_4P5_AIR_FP8 = auto()
    GLM_4P5 = auto()
    GLM_4P6 = auto()
    GLM_4P7 = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

client_mode_group = {
    "GLM-4.5-AIR-FP8":{
        "chat": ClientMode.GLM_4P5_AIR_FP8,
        "api_type": "together"
    },
    "glm-4p5":{
        "chat": ClientMode.GLM_4P5,
        "api_type": "fireworks"
    },
    "glm-4p7":{
        "chat": ClientMode.GLM_4P7,
        "api_type": "fireworks"
    },
    "GLM-4.6": {
        "chat": ClientMode.GLM_4P6,
        "api_type": "deepinfra"
    }
}

local_mode_group = {}

class ZhipuAILLM(AbstractLLM):
    """
    Class for models from z.ai
    """
    def __init__(self, config: ZhipuAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.GLM_4P5_AIR_FP8:
                    together_name = f"zai-org/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                        model=together_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens = self.max_tokens,
                        temperature = self.temperature
                    )
                    summary = response.choices[0].message.content
                case ClientMode.GLM_4P5:
                    self.model_fullname = f"accounts/fireworks/models/{self.model_name}"
                    response = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prepared_text,
                            }
                        ],
                        model=self.model_fullname,
                    )

                    summary = response.choices[0].message.content
                case ClientMode.GLM_4P7:
                    self.model_fullname = f"accounts/fireworks/models/{self.model_name}"
                    response = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prepared_text,
                            }
                        ],
                        model=self.model_fullname,
                    )

                    summary = response.choices[0].message.content

                case ClientMode.GLM_4P6:
                    self.model_fullname = f"{COMPANY}/{self.model_name}"
                    chat_completion = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content": prepared_text}],
                    )
                    summary = chat_completion.choices[0].message.content

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
            if self.model_name in client_mode_group:
                if client_mode_group[self.model_name]["api_type"] == "together":
                    api_key = os.getenv(f"TOGETHER_API_KEY")
                    assert api_key is not None, f"TOGETHER API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = Together(api_key=api_key)
                elif client_mode_group[self.model_name]["api_type"] == "fireworks":
                    api_key = os.getenv(f"FIREWORKS_API_KEY")
                    assert api_key is not None, f"FIREWORKS API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.fireworks.ai/inference/v1"
                    )
                elif client_mode_group[self.model_name]["api_type"] == "deepinfra":
                    api_key = os.getenv(f"DEEPINFRA_API_KEY")
                    assert api_key is not None, f"DEEPINFRA API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepinfra.com/v1/openai"
                    )
                else:
                    self.client  = None
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