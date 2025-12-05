import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
import json
import boto3

COMPANY = "amazon" #Official company name on huggingface
class AmazonConfig(BasicLLMConfig):
    """Extended config for amazon-specific properties"""
    company: Literal["amazon"] = "amazon"
    model_name: Literal[
        "nova-pro-v2",
        "nova-2-lite-v1:0",
        "nova-lite-v1:0",
        "nova-micro-v1:0",
        "nova-pro-v1:0",
    ] # Only model names manually added to this list are supported.
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class AmazonSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.

    class Config:
        extra = "ignore" 

class AmazonLLM(AbstractLLM):
    """
    Class for models from company_name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "nova-pro-v2": {
            "chat": 1
        },
        "nova-2-lite-v1:0": {
            "chat": 1
        },
        "nova-lite-v1:0": {
            "chat": 1
        },
        "nova-micro-v1:0": {
            "chat": 1
        },
        "nova-pro-v1:0": {
            "chat": 1
        },
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: AmazonConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config
        self.model_fullname = f"us.amazon.{self.model_name}"

    def summarize(self, prepared_text: str) -> str:
        # Use self.model_fullname when referring to the model
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1:
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
            if self.model_name in self.client_mode_group:
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
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass