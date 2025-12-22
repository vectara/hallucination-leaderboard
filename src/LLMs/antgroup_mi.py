import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

import json
import multiprocessing
import requests

COMPANY = "antgroup"

class AntGroupMIConfig(BasicLLMConfig):
    company: Literal["antgroup"] = "antgroup"
    model_name: Literal[
        "finix_s1_32b",
        "antfinix-a1"
    ]
    execution_mode: Literal["api"] = "api"
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"

class AntGroupMISummary(BasicSummary):
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
    "finix_s1_32b":{
        "chat": 1
    },
    "antfinix-a1":{
        "chat":2
    }
}

local_mode_group = {} 

class AntGroupMILLM(AbstractLLM):
    """
    Class for models from AntGroup-MI
    """
    def __init__(self, config: AntGroupMIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
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

                case 2:
                    base_url = "https://antfinix.alipay.com/v1/chat/completions"
                    messages = [{"role": "user", "content": prepared_text}]
                    summary = self.call_insllm_api_v2(
                        api_token=self.api_key,
                        base_url=base_url,
                        model_name="antfinix-a1",
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    pass
        elif self.local_model:
            pass
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
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
            pass
        elif self.local_model:
            pass

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

    def call_insllm_api_v2(
            self,
            api_token,
            base_url,
            model_name,
            messages,
            temperature,
            max_tokens
        ):
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {api_token}"
        }
        stream_state = True
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream_state,  # avoid timeout for long think and response
        }
        try:
            response = requests.post(base_url, headers=headers, json=payload, stream=stream_state)
            result = ""
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        json_str = line[5:]
                        if json_str.strip() == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                                delta = chunk["choices"][0]["delta"]
                                if "reasoning_content" in delta and delta["reasoning_content"]:
                                    pass
                                elif "content" in delta and delta["content"]:
                                    result += delta["content"]

                        except json.JSONDecodeError:
                            print(f"Error Decoding JSON: {json_str}")
            
            return result
        except Exception as e:
            print(f"Error calling INSLLM API: {e}")
            return "ERROR CALLING INSLLM API"