import os
import torch
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPANY = "01-ai"

class _01AIConfig(BasicLLMConfig):
    company: Literal["01-ai"] = "01-ai"
    model_name: Literal[
        "Yi-1.5-6B-Chat",
        "Yi-1.5-9B-Chat",
        "Yi-1.5-34B-Chat",
    ]
    date_code: str = "",
    endpoint: Literal["chat", "response"] = "chat"
    execution_mode: Literal["gpu", "cpu"] = "gpu"

class _01AISummary(BasicSummary):
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

client_mode_group = {} 

local_mode_group = {
    "Yi-1.5-6B-Chat": { 
        "chat": LocalMode.UNDEFINED
    },
    "Yi-1.5-9B-Chat": {
        "chat": LocalMode.CHAT_DEFAULT
    },
    "Yi-1.5-34B-Chat": {
        "chat": LocalMode.CHAT_DEFAULT
    }
}


class _01AILLM(AbstractLLM):
    """
    Class for models from 01-AI

    Attributes:

    """
    def __init__(self, config: _01AIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.model_fullname = f"{COMPANY}/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            pass
        elif self.local_model:
            match local_mode_group[self.model_name][self.endpoint]:
                case LocalMode.CHAT_DEFAULT: # Uses chat template
                    tokenizer = AutoTokenizer.from_pretrained(self.model_fullname, use_fast=False)

                    messages = [
                        {"role": "user", "content": prepared_text}
                    ]

                    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
                    output_ids = self.local_model.generate(
                        input_ids.to('cuda'),
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                    summary = response
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            pass
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
                # bnb_config = BitsAndBytesConfig(
                #     load_in_4bit=True,
                # )

                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.model_fullname,
                    device_map="auto",
                    torch_dtype="auto"
                ).to(self.device).eval()
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