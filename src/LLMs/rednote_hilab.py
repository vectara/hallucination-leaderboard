import os
import torch
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


COMPANY = "rednote-hilab"

class RednoteHilabConfig(BasicLLMConfig):
    company: Literal["rednote-hilab"] = "rednote-hilab"
    model_name: Literal["rednote-model"]
    execution_mode: Literal["api"] = "api"
    date_code: str

class RednoteHilabSummary(BasicSummary):
    pass

class ClientMode(Enum):
    DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

class LocalMode(Enum):
    DEFAULT = auto()
    # TODO: Add more as needed, make the term descriptive

client_mode_group = {}

local_mode_group = {
    "rednote-hilab/dots.llm1.inst": 1, # Uses chat template
    "rednote-hilab/dots.llm1.base": 2 # Uses direct text input
}

class RednoteHilabLLM(AbstractLLM):
    """
    Class for models from Rednote
    """
    def __init__(self, config: RednoteHilabConfig):
        super().__init__(config)

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            pass
        elif self.local_model:
            match local_mode_group[self.model_name]:
                case 1: # Uses chat template
                    tokenizer = AutoTokenizer.from_pretrained(self.model_fullname)

                    input_tensor = tokenizer.apply_chat_template(
                        {"role": "user", "content": prepared_text},
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )

                    outputs = self.local_model.generate(
                        input_tensor.to(self.local_model.device),
                        max_new_tokens=self.max_tokens
                    )

                    result = tokenizer.decode(
                        outputs[0][input_tensor.shape[1]:],
                        skip_special_tokens=True
                    )

                    summary = result
                case 2: # Uses direct text input
                    tokenizer = AutoTokenizer.from_pretrained(self.model_fullname)

                    inputs = tokenizer(prepared_text, return_tensors="pt")
                    outputs = self.local_model.generate(
                        **inputs.to(self.local_model.device),
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True
                    )
                    result = tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    summary = result
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            pass
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.model_fullname,
                    quantization_config=bnb_config
                ).to(self.device)
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