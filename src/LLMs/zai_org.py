import os
from typing import Literal
from together import Together
from openai import OpenAI


from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "zai-org"
class ZhipuAIConfig(BasicLLMConfig):
    company: Literal["zai-org"] = "zai-org" 
    model_name: Literal[
        "GLM-4.5-AIR-FP8", # Together
        "glm-4p5", # Fireworks but using OpenAI
        "glm-4-9b-chat",
        "GLM-4.6"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class ZhipuAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ZhipuAILLM(AbstractLLM):
    """
    Class for models from z.ai
    """

    client_mode_group = {
        "GLM-4.5-AIR-FP8":{
            "chat": 1,
            "api_type": "together"
        },
        "glm-4p5":{
            "chat": 2,
            "api_type": "fireworks"
        },
        "GLM-4.6": {
            "chat": 3,
            "api_type": "deepinfra"
        }
    }

    local_mode_group = {}

    def __init__(self, config: ZhipuAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Together API
                    together_name = f"zai-org/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                        model=together_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens = self.max_tokens,
                        temperature = self.temperature
                    )
                    summary = response.choices[0].message.content
                case 2: # Fireworks but using OpenAI
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

                case 3: # Deepinfra glm 4.6
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
            if self.model_name in self.client_mode_group:
                if self.client_mode_group[self.model_name]["api_type"] == "together":
                    api_key = os.getenv(f"TOGETHER_API_KEY")
                    assert api_key is not None, f"TOGETHER API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = Together(api_key=api_key)
                elif self.client_mode_group[self.model_name]["api_type"] == "fireworks":
                    api_key = os.getenv(f"FIREWORKS_API_KEY")
                    assert api_key is not None, f"FIREWORKS API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.fireworks.ai/inference/v1"
                    )
                elif self.client_mode_group[self.model_name]["api_type"] == "deepinfra":
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