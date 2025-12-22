import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
import json
import boto3

COMPANY = "amazon"
class AmazonConfig(BasicLLMConfig):
    company: Literal["amazon"] = "amazon"
    model_name: Literal[
        "nova-pro-v2",
        "nova-2-lite-v1:0",
        "nova-lite-v1:0",
        "nova-micro-v1:0",
        "nova-pro-v1:0",
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class AmazonSummary(BasicSummary):
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
    "nova-pro-v2": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "nova-2-lite-v1:0": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "nova-lite-v1:0": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "nova-micro-v1:0": {
        "chat": ClientMode.CHAT_DEFAULT
    },
    "nova-pro-v1:0": {
        "chat": ClientMode.CHAT_DEFAULT
    },
}

local_mode_group = {}

class AmazonLLM(AbstractLLM):
    """
    Class for models from amazon
    """
    def __init__(self, config: AmazonConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config
        self.model_fullname = f"us.amazon.{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
                    response_package = self.client.invoke_model(
                        modelId=self.model_fullname,
                        body=json.dumps({
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"text": prepared_text}
                                    ]
                                }
                            ],
                            "inferenceConfig": {
                                "temperature": self.temperature,
                                "maxTokens": self.max_tokens
                            }
                        })
                    )
                    raw = response_package["body"].read()
                    model_response = json.loads(raw)

                    summary = model_response["output"]["message"]["content"][0]["text"]
                    summary = summary.strip()
                    if summary.startswith('"') and summary.endswith('"'):
                        summary = summary[1:-1].strip()
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
                api_key = os.getenv(f"AWS_SECRET_ACCESS_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"AWS_SECRET_ACCESS_KEY"
                )
                self.client = boto3.client("bedrock-runtime", region_name="us-west-2")
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