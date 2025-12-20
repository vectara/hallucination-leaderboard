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
    DEFAULT_CHAT = auto()
    CHAT_NO_TEMP = auto()
    CHAT_NO_TEMP_NO_REASONING = auto()
    DEFAULT_RESPONSE = auto()
    RESPONSE_NO_TEMP = auto()
    DEFAULT_TOGETHER_API = auto()
    DEFAULT_REPLICATE_API = auto()
    O4_MINI_LOW = auto()
    O4_MINI_HIGH = auto()
    GPT_5P2_HIGH = auto()
    GPT_5P2_LOW = auto()
    GPT_5P1_HIGH = auto()
    GPT_5P1_LOW = auto()
    GPT_5_HIGH = auto()
    GPT_5_MINIMAL = auto()
    GPT_5_DEFAULT = auto()

class LocalMode(Enum):
    DEFAULT_CHAT = auto()

client_mode_group = {
    "gpt-5.2-low": {
        "chat": ClientMode.GPT_5P2_LOW,
        "api_type": "openai"
    },
    "gpt-5.2-high": {
        "chat": ClientMode.GPT_5P2_HIGH,
        "api_type": "openai"
    },
    "gpt-5.1-low": {
        "chat": ClientMode.GPT_5P1_LOW,
        "api_type": "openai"
    },
    "gpt-5.1-high": {
        "chat": ClientMode.GPT_5P1_HIGH,
        "api_type": "openai"
    },
    "gpt-5-minimal": {
        "chat": ClientMode.GPT_5_MINIMAL,
        "api_type": "openai"
    },
    "gpt-5-high": {
        "chat": ClientMode.GPT_5_HIGH,
        "api_type": "openai"
    },
    "gpt-5": {
        "chat": ClientMode.GPT_5_DEFAULT,
        "api_type": "openai"
    },
    "gpt-5-mini": {
        "chat": ClientMode.GPT_5_DEFAULT,
        "api_type": "openai"
    },
    "gpt-5-nano": {
        "chat": ClientMode.GPT_5_DEFAULT,
        "api_type": "openai",
    },
    "gpt-4.1": {
        "chat": ClientMode.DEFAULT_CHAT,
        "api_type": "openai",
        "response": ClientMode.DEFAULT_RESPONSE
    },
    "gpt-4.1-nano": {
        "chat": ClientMode.DEFAULT_CHAT,
        "api_type": "openai",
        "response": ClientMode.DEFAULT_RESPONSE
    },
    "o3": {
        "chat": ClientMode.CHAT_NO_TEMP,
        "api_type": "openai",
        "response": ClientMode.DEFAULT_RESPONSE
    },
    "o3-pro": {
        "api_type": "openai",
        "response": ClientMode.RESPONSE_NO_TEMP
    },
    "o4-mini": {
        "chat": ClientMode.CHAT_NO_TEMP,
        "api_type": "openai",
    },
    "o4-mini-low": {
        "chat": ClientMode.O4_MINI_LOW,
        "api_type": "openai",
    },
    "o4-mini-high": {
        "chat": ClientMode.O4_MINI_HIGH,
        "api_type": "openai",
    },
    "o1-pro": {
        "api_type": "openai",
        "response": ClientMode.RESPONSE_NO_TEMP
    },
    "gpt-4.1-mini": {
        "chat": ClientMode.DEFAULT_CHAT,
        "api_type": "openai",
    },
    "o1": {
        "chat": ClientMode.CHAT_NO_TEMP,
        "api_type": "openai",
    },
    "o1-mini": {
        "chat": ClientMode.CHAT_NO_TEMP_NO_REASONING,
        "api_type": "openai",
    },
    "gpt-oss-120b": {
        "chat": ClientMode.DEFAULT_TOGETHER_API,
        "api_type": "together"
    },
    "gpt-oss-20b": {
        "chat": ClientMode.DEFAULT_REPLICATE_API,
        "api_type": "replicate"
    },
    "gpt-4o-mini": {
        "chat": ClientMode.DEFAULT_CHAT,
        "api_type": "openai",
    },
    "gpt-4o": {
        "chat": ClientMode.DEFAULT_CHAT,
        "api_type": "openai",
    },
    "gpt-4-turbo": {
        "chat": ClientMode.DEFAULT_CHAT,
        "api_type": "openai",
    },
    "gpt-3.5-turbo": {
        "chat": ClientMode.DEFAULT_CHAT,
        "api_type": "openai",
    },
    "gpt-4": {
        "chat": ClientMode.DEFAULT_CHAT,
        "api_type": "openai",
    }
}

local_mode_group = {
    "gpt-oss-20b": {
        "chat": LocalMode.DEFAULT_CHAT
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
        if self.model_name in local_mode_group:
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
                case ClientMode.DEFAULT_CHAT:
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content":prepared_text}]
                    )
                    summary = chat_package.choices[0].message.content
                case ClientMode.CHAT_NO_TEMP:
                    chat_package = self.client.chat.completions.create(
                        model=self.model_fullname,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = self.reasoning_effort
                    )
                    summary = chat_package.choices[0].message.content
                case ClientMode.O4_MINI_LOW:
                    chat_package = self.client.chat.completions.create(
                        model="o4-mini-2025-04-16",
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = "low"
                    )
                    summary = chat_package.choices[0].message.content
                case ClientMode.O4_MINI_HIGH:
                    chat_package = self.client.chat.completions.create(
                        model="o4-mini-2025-04-16",
                        messages=[{"role": "user", "content":prepared_text}],
                        max_completion_tokens=self.max_tokens,
                        reasoning_effort = "high"
                    )
                    summary = chat_package.choices[0].message.content
                case ClientMode.GPT_5_DEFAULT:
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
                case ClientMode.GPT_5_MINIMAL:
                    chat_package = self.client.responses.create(
                        model="gpt-5-2025-08-07",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "minimal"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5_HIGH:
                    chat_package = self.client.responses.create(
                        model="gpt-5-2025-08-07",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5P1_LOW:
                    chat_package = self.client.responses.create(
                        model="gpt-5.1-2025-11-13",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "low"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5P1_HIGH:
                    chat_package = self.client.responses.create(
                        model="gpt-5.1-2025-11-13",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = chat_package.output[1].content[0].text
                case ClientMode.GPT_5P2_LOW:
                    chat_package = self.client.responses.create(
                        model="gpt-5.2-2025-12-11",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "low"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = self.extract_summary(chat_package)
                case ClientMode.GPT_5P2_HIGH:
                    chat_package = self.client.responses.create(
                        model="gpt-5.2-2025-12-11",
                        input=prepared_text,
                        max_output_tokens=self.max_tokens,
                        reasoning={
                            "effort": "high"
                        }
                    )
                    self.temperature = chat_package.temperature
                    summary = self.extract_summary(chat_package)
                case ClientMode.DEFAULT_TOGETHER_API:
                    together_name = f"openai/{self.model_fullname}"
                    response = self.client.chat.completions.create(
                        model=together_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        max_tokens = self.max_tokens,
                        temperature = self.temperature
                    )
                    summary = response.choices[0].message.content
                case ClientMode.DEFAULT_REPLICATE_API:
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
                case ClientMode.DEFAULT_RESPONSE:
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text,
                        reasoning = {"effort": self.reasoning_effort}
                    )
                    summary = chat_package.output_text
                case ClientMode.RESPONSE_NO_TEMP:
                    chat_package = self.client.responses.create(
                        model=self.model_fullname,
                        max_output_tokens=self.max_tokens,
                        input=prepared_text,
                        reasoning = {"effort": self.reasoning_effort}
                    )
                    summary = chat_package.output_text
                case ClientMode.CHAT_NO_TEMP_NO_REASONING:
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
                case LocalMode.DEFAULT_CHAT:
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
