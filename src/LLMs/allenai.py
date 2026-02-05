"""Allen AI (OLMo) model implementations for hallucination evaluation.

This module provides the LLM implementation for Allen AI's OLMo model family,
supporting multiple execution modes including API inference (via HuggingFace
or OpenRouter), local GPU inference, and vLLM for high-throughput serving.

Classes:
    AllenAIConfig: Configuration model for OLMo model settings.
    AllenAISummary: Output model for OLMo summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    AllenAILLM: Main LLM class implementing AbstractLLM for OLMo models.

Attributes:
    COMPANY: Provider identifier string ("allenai").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to supported local execution modes.
"""

import os
import torch
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import InferenceClient
from openai import OpenAI
# from vllm import SamplingParams
# from vllm import LLM
# from vllm.config import CompilationConfig, CompilationMode

COMPANY = "allenai"
"""str: Provider identifier used for model path construction and registration."""

class AllenAIConfig(BasicLLMConfig):
    """Configuration model for Allen AI OLMo models.

    Extends BasicLLMConfig with OLMo-specific settings for model selection
    and execution mode configuration. Supports multiple execution backends.

    Attributes:
        company: Provider identifier, fixed to "allenai".
        model_name: Name of the OLMo model variant to use. Includes both
            reasoning models (Think) and instruction-tuned models (Instruct).
        date_code: Optional version/date identifier inserted into model name.
        endpoint: Inference endpoint type ("chat" for conversational format).
        execution_mode: Where to run inference ("api", "gpu", "cpu", or "vllm").
    """

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
    api_type: Literal["huggingface", "openrouter"] | None = None  # Required for api mode

class AllenAISummary(BasicSummary):
    """Output model for Allen AI OLMo summarization results.

    Extends BasicSummary with endpoint and execution mode tracking
    for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
        execution_mode: The execution backend used for generation.
    """

    endpoint: Literal["chat", "response"] | None = None
    execution_mode: Literal["api", "gpu", "cpu", "vllm"] | None = None
    api_type: Literal["huggingface", "openrouter"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using an API client
    (HuggingFace Inference or OpenRouter).

    Attributes:
        CHAT_DEFAULT: Standard chat completion via HuggingFace Inference API.
        CHAT_REASONING: Chat completion with reasoning enabled (OpenRouter).
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    CHAT_REASONING = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally
    via HuggingFace transformers or vLLM.

    Attributes:
        CHAT_DEFAULT: Standard chat template inference on single GPU.
        CHAT_MGPU: Multi-GPU inference with automatic device mapping.
        CHAT_VLLM: High-throughput inference via vLLM engine.
        RESPONSE_DEFAULT: Direct completion without chat template.
        UNDEFINED: Mode not defined or not supported for this model.
    """

    CHAT_DEFAULT = auto()
    CHAT_MGPU = auto()
    CHAT_VLLM = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values and specifies the
# API provider ("hf" for HuggingFace, "openrouter" for OpenRouter).
client_mode_group = {
    "Olmo-3-32B-Think": {
        "chat": ClientMode.CHAT_REASONING
    },
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Each model maps endpoint types to LocalMode enum values indicating how to
# load and invoke the model locally.
local_mode_group = {
    "Olmo-3-32B-Think": {
        "chat": LocalMode.CHAT_VLLM,
    },
    "Olmo-3-7B-Think": {
        "chat": LocalMode.CHAT_VLLM,
    },
    "OLMo-2-7B-Instruct": {
        "chat": LocalMode.CHAT_DEFAULT
    },
    "OLMo-2-13B-Instruct": {
        "chat": LocalMode.CHAT_DEFAULT
    }
}

class AllenAILLM(AbstractLLM):
    """LLM implementation for Allen AI OLMo models.

    Provides text summarization using Allen AI's OLMo model family via multiple
    backends: API inference (HuggingFace or OpenRouter), local GPU inference
    with HuggingFace transformers, or high-throughput vLLM serving.

    Attributes:
        endpoint: The inference endpoint type (e.g., "chat").
        execution_mode: Where inference runs ("api", "gpu", "cpu", or "vllm").
        device: PyTorch device for local inference.
        model_fullname: Full model path including company prefix.
    """

    def __init__(self, config: AllenAIConfig):
        """Initialize the Allen AI LLM with the given configuration.

        Handles OLMo's special naming convention where the date code is
        inserted in the middle of the model name rather than at the end.

        Args:
            config: Configuration object specifying model and execution settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.api_type = config.api_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Olmo has it's date code in the middle
        if self.date_code != "":
            self.model_fullname = f"{self.model_name[0:6]}-{config.date_code}{self.model_name[6:]}"
        self.model_fullname = f"{COMPANY}/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured OLMo model to generate a condensed summary.
        Supports multiple inference modes: API client (HuggingFace/OpenRouter),
        local transformers, multi-GPU, and vLLM.

        Args:
            prepared_text: The preprocessed text to summarize.

        Returns:
            The generated summary text, or an error placeholder if generation fails.

        Raises:
            Exception: If neither client nor local_model is initialized.
        """
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT: # Standard chat completion
                    messages = [{"role": "user", "content":prepared_text}]
                    client_package = self.client.chat_completion(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    summary = client_package.choices[0].message.content
                case ClientMode.CHAT_REASONING:
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

                case LocalMode.CHAT_MGPU: # mpgu
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
                case LocalMode.CHAT_VLLM:  # vllm
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
        """Initialize the model for inference based on execution mode.

        For API mode, creates either a HuggingFace InferenceClient or OpenAI
        client (for OpenRouter) depending on the provider configuration.
        For vLLM mode, initializes a vLLM LLM engine with tensor parallelism.
        For GPU/CPU mode, loads the model via HuggingFace transformers with
        automatic device mapping and memory optimization.

        Raises:
            AssertionError: If the required API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                if self.api_type == "huggingface":
                    api_key = os.getenv("HF_TOKEN")
                    assert api_key is not None, "HF_TOKEN not found in environment variable HF_TOKEN"
                    self.client = InferenceClient(model=self.model_fullname, token=api_key)
                elif self.api_type == "openrouter":
                    api_key = os.getenv("OPENROUTER_API_KEY")
                    assert api_key is not None, "OPENROUTER_API_KEY not found in environment variable OPENROUTER_API_KEY"
                    self.client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_key,
                    )
                else:
                    raise ValueError(f"api_type must be 'huggingface' or 'openrouter' for API mode, got: {self.api_type}")
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode == "vllm":
            if self.model_name in local_mode_group:
                self.local_model = LLM(
                    model=self.model_fullname,
                    tensor_parallel_size=8,   # A100-80G x8
                    max_model_len=self.max_tokens,
                    # compilation_config=CompilationConfig( # customize graph capturing
                    #     mode=CompilationMode.VLLM_COMPILE,
                    #     # By default, it goes up to max_num_seqs
                    #     cudagraph_capture_sizes=[1, 2, 4, 8, 16],
                    # ),
                    enforce_eager=True, # Disables graph capturing
                )
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode in ["gpu", "cpu"]:
            if self.model_name in local_mode_group:
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
        """Clean up resources after inference is complete.

        Releases any held resources from the client or local model.
        Currently a no-op as cleanup is handled by garbage collection.
        """
        if self.client:
            return
        elif self.local_model:
            return

    def close_client(self):
        """Close any active API client connections.

        Currently a no-op as the API clients do not require explicit cleanup.
        """
        pass