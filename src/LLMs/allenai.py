import os
import torch
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import InferenceClient
from openai import OpenAI
from vllm import SamplingParams
from vllm import LLM

COMPANY = "allenai"

class AllenAIConfig(BasicLLMConfig):
    company: Literal["allenai"] = "allenai"
    model_name: Literal[
        "Olmo-3-32B-Think",
        "Olmo-3-7B-Think",
        "OLMo-2-7B-Instruct",
        "OLMo-2-13B-Instruct",
        "OLMo-2-0325-32B-Instruct",
        "OLMo-2-1124-7b-instruct",
        "OLMo-2-1124-13b-instruct",
    ]
    date_code: str = ""
    endpoint: Literal["chat", "response"] = "chat"
    execution_mode: Literal["api", "gpu", "cpu", "vllm"] = "api"

class AllenAISummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None
    execution_mode: Literal["api", "gpu", "cpu", "vllm"] | None = None

    class Config:
        extra = "ignore"

class AllenAILLM(AbstractLLM):
    """
    Class for models from AllenAI

    Attributes:
    """

    client_mode_group = {
        "Olmo-3-32B-Think": {
            "chat": 2,
            "provider": "openrouter"
        },
    }

    local_mode_group = {
        "Olmo-3-32B-Think": {
            "chat": 3,
        },
        "Olmo-3-7B-Think": {
            "chat": 3,
        },
        "OLMo-2-7B-Instruct": {
            "chat": 1
        },
        "OLMo-2-13B-Instruct": {
            "chat": 1
        }
    }

    def __init__(self, config: AllenAIConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Olmo has it's date code in the middle
        if self.date_code is not "":
            self.model_fullname = f"{self.model_name[0:6]}-{config.date_code}{self.model_name[6:]}"
        self.model_fullname = f"{COMPANY}/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Standard chat completion
                    messages = [{"role": "user", "content":prepared_text}]
                    client_package = self.client.chat_completion(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = client_package.choices[0].message.content
                case 2:
                    unique_name = f"{self.model_fullname}:free".lower()
                    messages = [{"role": "user", "content":prepared_text}]
                    response = self.client.chat.completions.create(
                    model=unique_name,
                    messages=messages,
                    extra_body={"reasoning": {"enabled": True}}
                    )
                    summary = response.choices[0].message.content
                    print(summary)
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
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                    summary = response

                case 2: # mpgu
                    tokenizer = AutoTokenizer.from_pretrained(self.model_fullname)
                    inputs = tokenizer(
                        prepared_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_tokens
                    ).to(self.local_model.device)

                    output = self.local_model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        num_beams=1,
                        do_sample=False
                    )

                    summary = tokenizer.decode(output[0], skip_special_tokens=True)
                case 3:  # vllm
                    sampling_params = SamplingParams(
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                    outputs = self.local_model.generate(
                        prepared_text,
                        sampling_params,
                    )

                    summary = outputs[0].outputs[0].text
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                if self.client_mode_group[self.model_name]["provider"] == "hf":
                    self.client = InferenceClient(model=self.model_fullname)
                elif self.client_mode_group[self.model_name]["provider"] == "openrouter":
                    api_key = os.getenv(f"OPENROUTER_API_KEY")
                    assert api_key is not None, f"{COMPANY} API key not found in environment variable {COMPANY.upper()}_API_KEY"
                    self.client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_key,
                    )
                else:
                    self.client = None
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode == "vllm":
            self.local_model = LLM(
                model=self.model_fullname,
                tensor_parallel_size=8,   # A100-80G x8
                dtype="float16",
                max_model_len=self.max_tokens,
            )
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in self.local_mode_group:
                max_memory = {
                    i: "64GiB" for i in range(torch.cuda.device_count())
                }

                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.model_fullname,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory=max_memory,
                    attn_implementation="sdpa",
                    low_cpu_mem_usage=True
                )



                # bnb_config = BitsAndBytesConfig(
                #     load_in_4bit=True,
                # )

                # self.local_model = AutoModelForCausalLM.from_pretrained(
                #     self.model_fullname,
                #     device_map="auto",
                #     torch_dtype="auto"
                # ).to(self.device).eval()
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))

    def teardown(self):
        if self.client:
            return
        elif self.local_model:
            return

    def close_client(self):
        pass