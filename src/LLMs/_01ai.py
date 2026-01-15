"""01.AI (Yi) model implementations for hallucination evaluation.

This module provides the LLM implementation for 01.AI's Yi model family,
supporting local GPU/CPU inference via HuggingFace transformers.

Classes:
    _01AIConfig: Configuration model for Yi model settings.
    _01AISummary: Output model for Yi summarization results.
    ClientMode: Enum for API client execution modes (currently unused).
    LocalMode: Enum for local model execution modes.
    _01AILLM: Main LLM class implementing AbstractLLM for Yi models.

Attributes:
    COMPANY: Provider identifier string ("01-ai").
    client_mode_group: Mapping of models to API client modes (currently empty).
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

COMPANY = "01-ai"
"""str: Provider identifier used for model path construction."""

class _01AIConfig(BasicLLMConfig):
    """Configuration model for 01.AI Yi models.

    Extends BasicLLMConfig with Yi-specific settings for model selection
    and execution mode configuration.

    Attributes:
        company: Provider identifier, fixed to "01-ai".
        model_name: Name of the Yi model variant to use.
        date_code: Optional version/date identifier for the model.
        endpoint: Inference endpoint type ("chat" for conversational format).
        execution_mode: Where to run inference ("gpu" or "cpu").
    """

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
    """Output model for 01.AI Yi summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using an API client.
    Currently unused as 01.AI models run locally.

    Attributes:
        CHAT_DEFAULT: Use the chat/conversational API endpoint.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally
    via HuggingFace transformers.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported for this model.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Currently empty as 01.AI models are executed locally.
client_mode_group = {}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Each model maps endpoint types to LocalMode enum values indicating how to
# format inputs and invoke the model.
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
    """LLM implementation for 01.AI Yi models.

    Provides text summarization using 01.AI's Yi model family via local
    HuggingFace transformers inference. Supports chat-formatted generation
    with configurable model variants.

    Attributes:
        endpoint: The inference endpoint type (e.g., "chat").
        execution_mode: Where inference runs ("gpu" or "cpu").
        model_fullname: Full HuggingFace model path (e.g., "01-ai/Yi-1.5-9B-Chat").
    """

    def __init__(self, config: _01AIConfig):
        """Initialize the 01.AI LLM with the given configuration.

        Args:
            config: Configuration object specifying model and execution settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.model_fullname = f"{COMPANY}/{self.model_fullname}"

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured Yi model to generate a condensed summary.
        Applies chat template formatting when using CHAT_DEFAULT mode.

        Args:
            prepared_text: The preprocessed text to summarize.

        Returns:
            The generated summary text, or an error placeholder if generation fails.

        Raises:
            Exception: If neither client nor local_model is initialized.
        """
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
        """Initialize the model for inference.

        Loads the Yi model from HuggingFace and prepares it for generation.
        For GPU/CPU modes, loads the model with automatic device mapping
        and sets it to evaluation mode.

        Raises:
            Exception: If the model does not support the configured execution mode.
        """
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
        """Clean up resources after inference is complete.

        Releases any held resources from the client or local model.
        Currently a no-op as cleanup is handled by garbage collection.
        """
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        """Close any active API client connections.

        Currently a no-op as 01.AI models run locally without an API client.
        """
        pass