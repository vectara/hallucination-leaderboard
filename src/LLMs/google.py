import os
from typing import Literal

from google import genai
from google.genai import types
from transformers import pipeline
import torch

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "google"

class GoogleConfig(BasicLLMConfig):
    """Extended config for Google-specific properties"""
    company: Literal["google"] = "google"
    model_name: Literal[
        "chat-bison-001",
        "flan-t5-large",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro-001",
        "gemini-2.0-flash-lite-preview",
        "gemini-2.0-pro-exp",
        "gemini-2.5-pro-exp",
        "text-bison-001",

        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash-preview", # 05-20
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemma-7b-it", # Try local
        "gemma-1.1-2b-it", # Try local
        "gemma-1.1-7b-it", # Try local
        "gemma-2-2b-it", # Try local
        "gemma-2-9b-it", # Try local
        "google/flan-t5-large" # Use through huggingface

    ] # Only model names manually added to this list are supported.
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().
    execution_mode: Literal["api", "gpu", "cpu"] = "api" # Google models can only be run via web api.
    date_code: str = "", # You must specify a date code for Google models.
    thinking_budget: Literal[-1, 0] = 0 # -1 is dynamic thinking, 0 thinking is off

class GoogleSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.
    thinking_budget: Literal[-1, 0] | None = None # -1 is dynamic thinking, 0 thinking is off

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class GoogleLLM(AbstractLLM):
    """
    Class for models from Google

    Attributes:
        client (genai.Client): client associated with api calls
        model_name (str): Google style model name
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution.
    client_mode_group = {
        "gemini-2.5-flash-preview": {
            "chat": 1
        }, # 05-20
        # "gemma-3-1b-it": {
        #     "chat": 1
        # },
        # "gemma-3-4b-it": {
        #     "chat": 1
        # },
        # "gemma-3-12b-it": {
        #     "chat": 1
        # },
        "gemma-3-27b-it": {
            "chat": 1
        },
        "gemini-2.0-pro-exp": {
            "chat": 1
        }, # 02-05
        "gemini-2.0-flash-001": {
            "chat": 1
        },
        "gemini-2.0-flash": {
            "chat": 1
        },
        "gemini-2.0-flash-exp": {
            "chat": 1
        },
        "gemini-2.0-flash-lite": {
            "chat": 1
        },
        "gemini-1.5-flash-002": {
            "chat": 1
        },
        "gemini-1.5-pro-002": {
            "chat": 1
        },
        "gemini-1.5-flash": {
            "chat": 1
        },
        "gemini-1.5-pro": {
            "chat": 1
        },
        "gemini-pro": {
            "chat": 1
        }, # Prob does not work
        "gemma-7n-it": {
            "chat": 1
        }, # Not officially listed
        "gemma-1.1-2b-it": {
            "chat": 1
        }, # Not officially listed
        "gemma-1.1-7b-it": {
            "chat": 1
        }, # Not officially listed
        "gemma-2-2b-it": {
            "chat": 1
        }, # Not officially listed
        "gemma-2-9b-it": {
            "chat": 1
        }, # Not officially listed
        "gemini-2.5-pro": {
            "chat": 2
        },
        "gemini-2.5-pro-preview": {
            "chat": 3
        },
        "gemini-2.5-flash-lite": {
            "chat": 4
        },
        "gemini-2.5-flash": {
            "chat": 4
        },
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {
        "gemma-3-1b-it": {
            "chat": 1
        },
        "gemma-3-4b-it": {
            "chat": 1
        },
        "gemma-3-12b-it": {
            "chat": 1
        },
    }

    def __init__(self, config: GoogleConfig):
        # Ensure that the parameters passed into the constructor are of the type GoogleConfig.
        
        # Call parent constructor to inherit all parent properties
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.thinking_budget = config.thinking_budget
        print("#################################")
        print(self.model_name)
        if self.model_name in self.local_mode_group:
            print("ADJUSTING MODEL NAME")
            self.model_fullname == f"{COMPANY}/{self.model_name}"
            print(self.model_fullname)


    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Default
                    response = self.client.models.generate_content(
                        model=self.model_fullname,
                        contents=prepared_text,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        ),
                    )
                    summary = response.text
                case 2: # gemini-2.5-pro - supports thinking tokens
                    response = self.client.models.generate_content(
                        model=self.model_fullname,
                        contents=prepared_text,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_tokens)
                        ),
                    )
                    summary = response.text
                case 3: # gemini-2.5-pro-preview - requires large output token amount
                    response = self.client.models.generate_content(
                        model=self.model_fullname,
                        contents=prepared_text,
                        config=types.GenerateContentConfig(
                            max_output_tokens=4096,
                            temperature=self.temperature
                        )
                    )
                    summary = response.text
                case 4: # gemini-2.5-flash-lite
                    response = self.client.models.generate_content(
                        model=self.model_fullname,
                        contents=prepared_text,
                        config=types.GenerateContentConfig(
                            max_output_tokens=self.max_tokens,
                            temperature=self.temperature,
                            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
                        )
                    )
                    summary = response.text
        elif self.local_model:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prepared_text}
                    ]
                }
            ]

            output = self.local_model(
                text=messages,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature
            )
            print(output[0]["generated_text"][-1]["content"])
        else:
            raise Exception(ModelInstantiationError.MISSING_SETUP.format(class_name=self.__class__.__name__))
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_GEMINI_API_KEY")
                assert api_key is not None, f"Google Gemini API key not found in environment variable {COMPANY.upper()}_GEMINI_API_KEY"
                self.client = genai.Client(api_key=api_key)
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode in ["gpu", "cpu"]:
            print(self.model_fullname)
            self.local_mode = pipeline(
                "text-to-text",
                model=self.model_fullname,
                device="cuda",
                torch_dtype=torch.bfloat16
            )

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass # Google models cannot be run locally.

    def close_client(self):
        pass
