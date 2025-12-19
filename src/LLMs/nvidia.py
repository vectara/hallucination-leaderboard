import os
from typing import Literal
import re
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

import replicate
from openai import OpenAI

COMPANY = "nvidia"
class NvidiaConfig(BasicLLMConfig):
    company: Literal["nvidia"] = "nvidia"
    model_name: Literal[
        "Nemotron-3-Nano-30B-A3B",
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class NvidiaSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

class LocalMode(Enum):
    DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

client_mode_group = {
    "Nemotron-3-Nano-30B-A3B": { 
        "chat": 2
    }
}

# TODO: Add local models here and specify what logic path to run that model
local_mode_group = {
    "MODEL_NAME": {
        "chat": 1
    }
} 

class NvidiaLLM(AbstractLLM):
    """
    Class for models from Nvidia
    """
    def __init__(self, config: NvidiaConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        def strip_thinking(text: str) -> str:
            # Remove <thinking>...</thinking> or <think>...</think>
            text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
            return text.strip()
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    replicate_name = f"{COMPANY}/{self.model_fullname}:135b4a9c545002830563436c88ea56b401d135faa59da6773bc5934d2ae56344"
                    output = replicate.run(
                        replicate_name,
                        input={
                            "prompt": prepared_text,
                            "temperature": self.temperature,
                            "max_new_tokens": self.max_tokens,
                            "enable_thinking": True,
                        }
                    )
                    chunks = []

                    for item in output:
                        chunks.append(item if isinstance(item, str) else item.get("text", ""))

                    raw_text = "".join(chunks)
                    summary = strip_thinking(raw_text)
                case 2:
                    deep_infra_name = f"{COMPANY}/{self.model_fullname}"
                    chat_completion = self.client.chat.completions.create(
                        model=deep_infra_name,
                        messages=[{"role": "user", "content": prepared_text}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = chat_completion.choices[0].message.content.lstrip()
        elif self.local_model: 
            match local_mode_group[self.model_name][self.endpoint]:
                # TODO Define how the case 1 model will run
                case 1:
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
                api_key = os.getenv(f"DEEPINFRA_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                )

                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepinfra.com/v1/openai",
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
            if self.model_name in local_mode_group:
                # TODO: Assign a local model if using a local model
                self.local_model = None
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )

    def teardown(self):
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass