import os
from typing import Literal
from enum import Enum, auto

from openai import OpenAI
from transformers import pipeline
from together import Together
import replicate

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "openai"

class OpenAIConfig(BasicLLMConfig):
    company: Literal["openai"] = "openai"
    model_name: Literal[
        "chatgpt-4o",
        "gpt-4.5-preview",
        "o1-preview",

        "gpt-5.2-high",
        "gpt-5.2-low",
        "gpt-5.1-high",
        "gpt-5.1-low",
        "gpt-5-high",
        "gpt-5-minimal",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-oss-120b",
        "gpt-oss-20b",
        "gpt-4.1",
        "gpt-4.1-nano",
        "o3",
        "o3-pro",
        "o4-mini",
        "o4-mini-low",
        "o4-mini-high",
        "o1-pro",
        "gpt-4.1-mini",
        "o1",
        "o1-mini",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-4"
    ]
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = None

class OpenAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

class LocalMode(Enum):
    DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

client_mode_group = {
    "gpt-5.2-low": {
        "chat": 15,
        "api_type": "openai"
    },
    "gpt-5.2-high": {
        "chat": 16,
        "api_type": "openai"
    },
    "gpt-5.1-low": {
        "chat": 13,
        "api_type": "openai"
    },
    "gpt-5.1-high": {
        "chat": 14,
        "api_type": "openai"
    },
    "gpt-5-minimal": {
        "chat": 10,
        "api_type": "openai"
    },
    "gpt-5-high": {
        "chat": 11,
        "api_type": "openai"
    },
    "gpt-5": {
        "chat": 9,
        "api_type": "openai"
    },
    "gpt-5-mini": {
        "chat": 9,
        "api_type": "openai"
    },
    "gpt-5-nano": {
        "chat": 9,
        "api_type": "openai",
    },
    "gpt-4.1": {
        "chat": 1,
        "api_type": "openai",
        "response": 3
    },
    "gpt-4.1-nano": {
        "chat": 1,
        "api_type": "openai",
        "response": 3
    },
    "o3": {  # o3 does not support temperature
        "chat": 2,
        "api_type": "openai",
        "response": 3
    },
    "o3-pro": { # o3-pro doesn't support chatting protocol or temperature
        "chat": None, 
        "api_type": "openai",
        "response": 4
    },
    "o4-mini": {
        "chat": 2,
        "api_type": "openai",
    },
    "o4-mini-low": {
        "chat": 6,
        "api_type": "openai",
    },
    "o4-mini-high": {
        "chat": 7,
        "api_type": "openai",
    },
    "o1-pro": { # doesn't support chatting or temperature
        "chat": None,
        "api_type": "openai",
        "response": 4
    },
    "gpt-4.1-mini": {
        "chat": 1,
        "api_type": "openai",
    },
    "o1": {
        "chat": 2,
        "api_type": "openai",
    },
    "o1-mini": { # Doesn't support reasoning effort or temperature
        "chat": 5,
        "api_type": "openai",
    },
    "gpt-oss-120b": {
        "chat": 8,
        "api_type": "together"
    },
    "gpt-oss-20b": {
        "chat": 12,
        "api_type": "replicate"
    },
    "gpt-4o-mini": {
        "chat": 1,
        "api_type": "openai",
    },
    "gpt-4o": {
        "chat": 1,
        "api_type": "openai",
    },
    "gpt-4-turbo": {
        "chat": 1,
        "api_type": "openai",
    },
    "gpt-3.5-turbo": {
        "chat": 1,
        "api_type": "openai",
    },
    "gpt-4": {
        "chat": 1,
        "api_type": "openai",
    }
}

local_mode_group = {
    "gpt-oss-20b": {
        "chat": 1
    },
}

class OpenAILLM(AbstractLLM):
    """
    Class for models from OpenAI
    """
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.reasoning_effort = config.reasoning_effort
        if self.model_name in self.local_mode_group:
            self.model_fullname = f"openai/{self.model_fullname}"

    def extract_summary(self, resp):
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                if getattr(item, "content", None):
                    for c in item.content:
                        if getattr(c, "text", None):
                            return c.text
        return ""

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case 1: # Chat with temperature and max_tokens
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content":prepared_text}]
                    )
                    summary = chat_package.choices[0].message.content
                case 2: # Chat without temperature
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = self.reasoning_effort
                    )
                    summary = chat_package.choices[0].message.content
                case 6: # o4-mini-low
                    chat_package = self.client.chat.completions.create(
                        model="o4-mini-2025-04-16", # need to talk about this case
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = "low"
                    )
                    summary = chat_package.choices[0].message.content
                case 7: # o4-mini-high
                    chat_package = self.client.chat.completions.create(
                        model="o4-mini-2025-04-16", # need to talk about this case
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = "high"
                    )
                    summary = chat_package.choices[0].message.content
                case 9: #gpt-5
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": self.reasoning_effort
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case 10: # gpt-5-minimal
                    chat_package = self.client.responses.create(
                        model="gpt-5-2025-08-07", # need to talk about this case
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "minimal"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case 11: # gpt-5-high, TODO: add reponse data at end from usage
                    chat_package = self.client.responses.create(
                        model="gpt-5-2025-08-07", # need to talk about this case
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case 13: # gpt-5.1-low
                    chat_package = self.client.responses.create(
                        model="gpt-5.1-2025-11-13", # need to talk about this case
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "low"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case 14: # gpt-5.1-high
                    chat_package = self.client.responses.create(
                        model="gpt-5.1-2025-11-13", # need to talk about this case
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case 15: # gpt-5.2-low
                    chat_package = self.client.responses.create(
                        model="gpt-5.2-2025-12-11", # need to talk about this case
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "low"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = self.extract_summary(chat_package)
                case 16: # gpt-5.2-high
                    chat_package = self.client.responses.create(
                        model="gpt-5.2-2025-12-11", # need to talk about this case
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = self.extract_summary(chat_package)
                case 8: # gpt-oss-120b not supported on open ai and too big to run locally, using together
                    together_name = f"openai/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                        model=together_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens = self.max_tokens,
                        temperature = self.temperature
                    )
                    summary = response.choices[0].message.content
                case 12: # manual 20b
                    input = {
                        "prompt": prepared_text,
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                    }
                    summary = replicate.run(
                        f"{COMPANY}/{self.model_name}",
                        input=input
                    )
                    summary = summary[0]
                case 3: # Use OpenAI's Response API
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text,
                        reasoning = {"effort": self.reasoning_effort}
                    )
                    summary = chat_package.output_text
                case 4: # Use OpenAI's Response API no temp
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text,
                        reasoning = {"effort": self.reasoning_effort}
                    )
                    summary = chat_package.output_text
                case 5: # Chat without temperature and reasoning effort
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                    )
                    summary = chat_package.choices[0].message.content
                case None:
                    raise Exception(f"Model `{self.model_name}` cannot be run from `{self.endpoint}` endpoint")
        elif self.local_model:
            match local_mode_group[self.model_name][self.endpoint]:
                case 1: # Chat with temperature and max_tokens
                    def extract_after_assistant_final(text):
                        keyword = "assistantfinal"
                        index = text.find(keyword)
                        if index != -1:
                            return text[index + len(keyword):].strip()
                        return ""  # Return empty string if keyword not found
                    messages = [
                        {"role": "user", "content": prepared_text},
                    ]

                    outputs = self.local_model(
                        messages,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    raw_text = outputs[0]["generated_text"][-1]["content"]
                    summary = extract_after_assistant_final(raw_text)
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if client_mode_group[self.model_name]["api_type"] == "together":
                    api_key = os.getenv(f"TOGETHER_API_KEY")
                    assert api_key is not None, f"TOGETHER API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = Together(api_key=api_key)
                if client_mode_group[self.model_name]["api_type"] == "replicate":
                    api_key = os.getenv(f"REPLICATE_API_TOKEN")
                    assert api_key is not None, f"REPLICATE API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = "replicate has no client"
                else:
                    api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                    assert api_key is not None, f"OpenAI API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(api_key=api_key)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
                self.local_model = pipeline(
                    "text-generation",
                    model=self.model_fullname,
                    torch_dtype="auto",
                    device_map="auto", # Set gpu?
                )
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))

    def teardown(self):
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass
