import os
import torch
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

# Import the Python package for the specific provider.
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPANY = "rednote"

class RednoteConfig(BasicLLMConfig):
    """Extended config for Rednote-specific properties"""
    company: Literal["rednote"] = "rednote"
    model_name: Literal["rednote-model"] # Only model names manually added to this list are supported.
    execution_mode: Literal["api"] = "api" # Rednote models can only be run via web api.
    date_code: str # You must specify a date code for Rednote models.

class RednoteSummary(BasicSummary):
    pass # Nothing additional to the BasicSummary class.

class RednoteJudgment(BasicJudgment):
    pass # Rednote does not have fields beyond BasicJudgment.

class RednoteLLM(AbstractLLM):
    """
    Class for models from Rednote

    Attributes:
        local_model (AutoModelForCausalLM): local model for inference
        model_name (str): Rednote style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {} # Empty for Rednote models because they cannot be run via web api.

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {
        "rednote-hilab/dots.llm1.inst": 1, # Uses chat template
        "rednote-hilab/dots.llm1.base": 2 # Uses direct text input
    }

    def __init__(self, config: RednoteConfig):
        # Ensure that the parameters passed into the constructor are of the type RednoteConfig.
        
        # Call parent constructor to inherit all parent properties
        super().__init__(config)

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            pass # Rednote models cannot be run via web api.
        elif self.local_model:
            match self.local_mode_group[self.model_name]:
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
            pass # Rednote models cannot be run via web api.
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in self.local_mode_group:
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
            self.close_client()
        elif self.local_model:
            self.default_local_model_teardown()

    def close_client(self):
        pass