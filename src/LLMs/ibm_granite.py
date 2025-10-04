import os
import torch
from typing import Literal

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError
import re
import gc

# Import the Python package for the specific provider.
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPANY = "ibm-granite"

class IBMGraniteConfig(BasicLLMConfig):
    """Extended config for IBM-specific properties"""
    company: Literal["ibm-granite"] = "ibm-granite"
    model_name: Literal[
        "granite-4.0-h-small",
        "granite-4.0-h-tiny",
        "granite-4.0-h-micro",
        "granite-4.0-micro",
        "granite-3.2-8b-instruct",
        "granite-3.2-2b-instruct",
        "granite-3.1-8b-instruct",
        "granite-3.1-2b-instruct",
        "granite-3.0-8b-instruct",
        "granite-3.0-2b-instruct"
    ] # Only model names manually added to this list are supported.
    endpoint: Literal["chat", "response"] = "chat"
    execution_mode: Literal["gpu", "cpu"] = "gpu"

class IBMGraniteSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class IBMGraniteLLM(AbstractLLM):
    """
    Class for models from IBM

    Attributes:
        local_model (AutoModelForCausalLM): local model for inference
        model_name (str): Rednote style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {} # Empty for Rednote models because they cannot be run via web api.

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {
        "granite-4.0-h-small": {
            "chat": 1,
            "multi-gpu": False
        },
        "granite-4.0-h-tiny": {
            "chat": 1,
            "multi-gpu": False
        },
        "granite-4.0-h-micro": {
            "chat": 2,
            "multi-gpu": False
        },
        "granite-4.0-micro": {
            "chat": 2,
            "multi-gpu": False
        },
        "granite-3.2-8b-instruct": {
            "chat": 1
        },
        "granite-3.2-2b-instruct": {
            "chat": 1
        },
        "granite-3.1-8b-instruct": {
            "chat": 1
        },
        "granite-3.1-2b-instruct": {
            "chat": 1
        },
        "granite-3.0-8b-instruct": {
            "chat": 1
        },
        "granite-3.0-2b-instruct": {
            "chat": 1
        }
    }

    def __init__(self, config: IBMGraniteConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_fullname = f"{COMPANY}/{self.model_fullname}"

        # self.model_path = config.model_path

    def manual_forward(self, input_ids, attention_mask=None):
        n_gpus = torch.cuda.device_count()
        device_ids = [f'cuda:{i}' for i in range(n_gpus)]

        # Move input_ids and attention_mask to first GPU
        input_ids = input_ids.to(device_ids[0])
        if attention_mask is not None:
            attention_mask = attention_mask.to(device_ids[0])

        # Embeddings
        hidden_states = self.local_model.transformer.wte(input_ids) + self.local_model.transformer.wpe(
            torch.arange(input_ids.size(1), device=device_ids[0])
        )

        # Pass through transformer layers on respective GPUs
        num_layers = len(self.local_model.transformer.h)
        layers_per_gpu = num_layers // n_gpus
        for i in range(n_gpus):
            start = i * layers_per_gpu
            end = (i + 1) * layers_per_gpu if i != n_gpus - 1 else num_layers

            hidden_states = hidden_states.to(device_ids[i])
            for layer in self.local_model.transformer.h[start:end]:
                # Handle attention_mask if needed
                if attention_mask is not None:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
                else:
                    hidden_states = layer(hidden_states)[0]

        # Move hidden_states to lm_head GPU
        hidden_states = hidden_states.to(device_ids[-1])
        logits = self.local_model.lm_head(hidden_states)

        return logits


    def summarize(self, prepared_text: str) -> str:
        def extract_assistant_response(text: str) -> str:
            pattern = r"<\|start_of_role\|>assistant<\|end_of_role\|>(.*?)<\|end_of_text\|>"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "FAILED TO FIND TEXT"
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            pass
        elif self.local_model:
            match self.local_mode_group[self.model_name][self.endpoint]:
                case 1: # Uses chat template
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_fullname,
                        use_fast=False,
                        return_attention_mask=True
                    )

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

                case 2: # micro has some issues with original method
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_fullname,
                        return_attention_mask=False
                    )

                    messages = [
                        {"role": "user", "content": prepared_text}
                    ]

                    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    input_tokens = tokenizer(chat, return_tensors="pt").to(self.device)
                    output = self.local_model.generate(
                        **input_tokens, 
                        max_new_tokens=self.max_tokens,
                        use_cache=False
                    )
                    output = tokenizer.batch_decode(output)
                    summary = extract_assistant_response(output[0])
                case 3:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_fullname,
                        use_fast=False,
                        return_attention_mask=True
                    )
                    messages = [
                        {"role": "user", "content": prepared_text}
                    ]
                    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt').input_ids

                    logits = self.manual_forward(input_ids)
                    predicted_ids = torch.argmax(logits, dim=-1).cpu()

                    response = tokenizer.decode(predicted_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                    summary = response

                case 4:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_fullname,
                        return_attention_mask=False
                    )
                    messages = [
                        {"role": "user", "content": prepared_text}
                    ]
                    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    input_tokens = tokenizer(chat, return_tensors="pt")

                    logits = self.manual_forward(input_tokens.input_ids)

                    predicted_ids = torch.argmax(logits, dim=-1).cpu()
                    output_text = tokenizer.batch_decode(predicted_ids)
                    summary = extract_assistant_response(output_text[0])

        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            pass
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in self.local_mode_group:
                # bnb_config = BitsAndBytesConfig(
                #     load_in_8bit=True,                       # ← Enable 8-bit loading
                #     load_in_8bit_fp32_cpu_offload=True       # ← Offload some weights to CPU
                # )

                # self.local_model = AutoModelForCausalLM.from_pretrained(
                #     self.model_fullname,
                #     device_map="auto",
                #     dtype="auto",               # Still used for compute, required by transformers
                #     quantization_config=bnb_config           # ← Pass quantization config
                # ).eval()
                if self.local_mode_group[self.model_name]["multi-gpu"]:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_fullname,
                        torch_dtype="auto"
                    )
                    
                    n_gpus = torch.cuda.device_count()
                    device_ids = [f'cuda:{i}' for i in range(n_gpus)]

                    # Example assumes model.transformer.h holds layers
                    num_layers = len(model.transformer.h)
                    layers_per_gpu = num_layers // n_gpus

                    # Move embeddings to first GPU
                    model.transformer.wte.to(device_ids[0])
                    model.transformer.wpe.to(device_ids[0])
                    model.lm_head.to(device_ids[-1])  # Output head on last GPU

                    # Move layers to GPUs manually
                    for i in range(n_gpus):
                        start = i * layers_per_gpu
                        end = (i + 1) * layers_per_gpu if i != n_gpus - 1 else num_layers
                        for layer in model.transformer.h[start:end]:
                            layer.to(device_ids[i])

                    self.local_model = model.eval()

                else:
                    self.local_model = AutoModelForCausalLM.from_pretrained(
                        self.model_fullname,
                        device_map="auto",
                        dtype="auto",
                        offload_folder="./offload",
                        low_cpu_mem_usage=True
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