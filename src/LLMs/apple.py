"""Apple (OpenELM) model implementations for hallucination evaluation.

This module provides the LLM implementation for Apple's OpenELM model family,
supporting local GPU/CPU inference via HuggingFace transformers.

Classes:
    AppleConfig: Configuration model for OpenELM model settings.
    AppleSummary: Output model for OpenELM summarization results.
    ClientMode: Enum for API client execution modes (currently unused).
    LocalMode: Enum for local model execution modes.
    AppleLLM: Main LLM class implementing AbstractLLM for OpenELM models.

Attributes:
    COMPANY: Provider identifier string ("apple").
    client_mode_group: Mapping of models to API client modes (empty).
    local_mode_group: Mapping of models to supported local execution modes.
"""

import os
from typing import Literal
import torch
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPANY = "apple"
"""str: Provider identifier used for model path construction and registration."""


class AppleConfig(BasicLLMConfig):
    """Configuration model for Apple OpenELM models.

    Extends BasicLLMConfig with OpenELM-specific settings for model selection
    and execution mode configuration.

    Attributes:
        company: Provider identifier, fixed to "apple".
        model_name: Name of the OpenELM model variant to use.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference ("gpu", "cpu", or "api").
        endpoint: Inference endpoint type ("chat" for conversational format).
    """

    company: Literal["apple"] = "apple"
    model_name: Literal[
        "OpenELM-3B-Instruct",
    ]
    date_code: str = ""
    execution_mode: Literal["api", "cpu", "gpu"] = "gpu"
    endpoint: Literal["chat", "response"] = "chat"

class AppleSummary(BasicSummary):
    """Output model for Apple OpenELM summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The inference endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using an API client.
    Currently unused as OpenELM models run locally.

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
        CHAT_DEFAULT: Use chat template formatting with Llama-2 tokenizer.
        RESPONSE_DEFAULT: Use direct text input without chat template.
        UNDEFINED: Mode not defined or not supported for this model.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Empty dict indicates OpenELM models do not support API execution.
client_mode_group = {}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Each model maps endpoint types to LocalMode enum values indicating how to
# format inputs and invoke the model locally.
local_mode_group = {
    "OpenELM-3B-Instruct": {
        "chat": LocalMode.CHAT_DEFAULT
    }
}

class AppleLLM(AbstractLLM):
    """LLM implementation for Apple OpenELM models.

    Provides text summarization using Apple's OpenELM model family via local
    HuggingFace transformers inference. Uses Llama-2 tokenizer for chat
    template formatting due to OpenELM's compatibility requirements.

    Attributes:
        endpoint: The inference endpoint type (e.g., "chat").
        execution_mode: Where inference runs ("gpu" or "cpu").
        full_config: Complete configuration object for reference.
        model_fullname: Full HuggingFace model path (e.g., "apple/OpenELM-3B-Instruct").
        device: PyTorch device for inference.
    """

    def __init__(self, config: AppleConfig):
        """Initialize the Apple LLM with the given configuration.

        Args:
            config: Configuration object specifying model and execution settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config
        self.model_fullname = f"{COMPANY}/{self.model_name}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the configured OpenELM model to generate a condensed summary.
        For chat mode, uses Llama-2 tokenizer's chat template for input formatting.

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
                case 1:
                    summary = None
        elif self.local_model: 
            match local_mode_group[self.model_name][self.endpoint]:
                case LocalMode.CHAT_DEFAULT: # Uses chat template
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
        """Initialize the model for inference.

        For GPU/CPU modes, loads the OpenELM model from HuggingFace with
        automatic device mapping and sets it to evaluation mode. Requires
        trust_remote_code=True for OpenELM's custom architecture.

        Raises:
            AssertionError: If API mode is used and the API key is not set.
            Exception: If the model does not support the configured execution mode.
        """
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

        Currently a no-op as OpenELM models run locally without an API client.
        """
        pass