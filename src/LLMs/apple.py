import os
from typing import Literal
import torch
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPANY = "apple"
class AppleConfig(BasicLLMConfig):
    company: Literal["apple"] = "apple"
    model_name: Literal[
        "OpenELM-3B-Instruct",
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "gpu"
    endpoint: Literal["chat", "response"] = "chat"

class AppleSummary(BasicSummary):
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
}

local_mode_group = {
    "OpenELM-3B-Instruct": {
        "chat": 1
    }
}

class AppleLLM(AbstractLLM):
    """
    Class for models from apple
    """
    def __init__(self, config: AppleConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config
        self.model_fullname = f"{COMPANY}/{self.model_name}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case 1:
                    summary = None
        elif self.local_model: 
            match local_mode_group[self.model_name][self.endpoint]:
                case 1: # Uses chat template
                    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=True)

                    messages = [
                        {"role": "user", "content": prepared_text}
                    ]

                    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
                    output_ids = self.local_model.generate(
                        input_ids.to(self.device),
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        use_cache=False
                    )
                    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                    summary = response

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
            raise Exception(
                ModelInstantiationError.MISSING_SETUP.format(
                    class_name=self.__class__.__name__
                )
            )
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = None
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.model_fullname,
                    device_map="auto",
                    dtype="auto",
                    trust_remote_code=True
                ).eval()
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