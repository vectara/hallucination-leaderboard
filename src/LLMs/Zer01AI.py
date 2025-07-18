import os
import torch
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

# Import the Python package for the specific provider.
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPANY = "01-ai"

class Zer01AIConfig(BasicLLMConfig):
    """Extended config for 01-AI-specific properties"""
    company: Literal["01-ai"] = "01-ai"
    model_name: Literal[
        "Yi-1.5-9B-Chat",
        "Yi-1.5-34B-Chat"
    ] # Only model names manually added to this list are supported.
    endpoint: Literal["chat", "response"] = "chat"
    execution_mode: Literal["gpu", "cpu"] = "gpu"

class Zer01AISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class Zer01AILLM(AbstractLLM):
    """
    Class for models from 01-AI

    Attributes:
        local_model (AutoModelForCausalLM): local model for inference
        model_name (str): Rednote style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {} # Empty for Rednote models because they cannot be run via web api.

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {
        "Yi-1.5-9B-Chat": {
            "chat": 1
        },
        "Yi-1.5-34B-Chat": {
            "chat": 1
        }
    }

    def __init__(self, config: Zer01AIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_fullname = f"{COMPANY}/{self.model_fullname}"

        # self.model_path = config.model_path

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            pass # Rednote models cannot be run via web api.
        elif self.local_model:
            match self.local_mode_group[self.model_name][self.endpoint]:
                case 1: # Uses chat template
                    tokenizer = AutoTokenizer.from_pretrained(self.model_fullname, use_fast=False)

                    messages = [
                        {"role": "user", "content": prepared_text}
                    ]

                    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
                    output_ids = self.local_model.generate(
                        input_ids.to('cuda'),
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens
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
            if self.model_name in self.local_mode_group:
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
            self.close_client()
        elif self.local_model:
            self.default_local_model_teardown()

    def close_client(self):
        pass