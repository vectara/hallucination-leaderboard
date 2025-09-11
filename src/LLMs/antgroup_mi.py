import os
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

import json
import multiprocessing
import requests

COMPANY = "antgroup"

class AntGroupMIConfig(BasicLLMConfig):
    """Extended config for AntGroup-MI-specific properties"""
    company: Literal["antgroup"] = "antgroup"
    model_name: Literal[
        "finix_s1_32b"
    ] # Only model names manually added to this list are supported.
    execution_mode: Literal["api"] = "api" # MistralAI models can only be run via web api.
    date_code: str = "" # You must specify a date code for MistralAI models.
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().

class AntGroupMISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore" 

class AntGroupMILLM(AbstractLLM):
    """
    Class for models from AntGroup-MI

    Attributes:
        client (): client associated with api calls
        model_name (str): MistralAI style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {
        "finix_s1_32b":{
            "chat": 1
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {} # Empty for MistralAI models because they cannot be run locally.

    def __init__(self, config: AntGroupMIConfig):
        # Ensure that the parameters passed into the constructor are of the type MistralAIConfig.
        
        # Call parent constructor to inherit all parent properties
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Standard chat completion
                    base_url = "https://antfinix.alipay.com/v1/chat/completions"
                    messages = [{"role": "user", "content": prepared_text}]
                    summary = self.call_insllm_api(
                        api_token=self.api_key,
                        base_url=base_url,
                        model_name="antfinix-ir1",
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                model_id = COMPANY.replace("-", "_")
                self.api_key = os.getenv(f"{model_id.upper()}_API_KEY")
                assert self.api_key is not None, f"{COMPANY} API key not found in environment variable {model_id}"
                self.client = True
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
            pass # MistralAI models cannot be run locally.

    def close_client(self):
        pass

    def call_insllm_api(
            self,
            api_token,
            base_url,
            model_name,
            messages,
            temperature,
            max_tokens
        ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,  # avoid timeout for long think and response
        }
        try:
            response = requests.post(base_url, headers=headers, json=payload, timeout=600)
            completion = ""
            for line in response.iter_lines():
                if line:
                    # remove "data: " prefix and parse JSON
                    line_text = line.decode("utf-8")
                    if line_text.startswith("data: "):
                        json_str = line_text[6:]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                content = chunk["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    completion += content
                        except json.JSONDecodeError:
                            print(f"Error Decoding JSON: {json_str}")
            
            return completion
        except Exception as e:
            print(f"Error calling INSLLM API: {e}")
            return None