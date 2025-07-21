from typing import Dict, Any

"""
This package contains implementations for various LLM providers.
The package provides a unified interface through the AbstractLLM base class.

Key functions in the AbstractLLM class:
- setup(): Initialize an LLM at the beginning of `with` block
- teardown(): Clean up an LLM at the end of `with` block
- summarize(): Summarize a given text
"""

# Disabled because it is dangerous. -- Forrest, 2025-07-03
# LLM_dir = os.path.dirname(__file__)
# for file in os.listdir(LLM_dir):
#     if file.endswith(".py") and file != "__init__.py":
#         file_name = file[:-3]
#         importlib.import_module(f"src.LLMs.{file_name}")


from . AbstractLLM import AbstractLLM

# All vendor classes are called _{Vendor}_Class below to avoid name conflicts. -- Forrest, 2025-07-03
from . openai import OpenAILLM, OpenAIConfig, OpenAISummary
from . anthropic import AnthropicLLM, AnthropicConfig, AnthropicSummary
from . qwen import AlibabaLLM, AlibabaConfig, AlibabaSummary
from . google import GoogleLLM, GoogleConfig, GoogleSummary
from . deepseek_ai import DeepSeekAILLM, DeepSeekAIConfig, DeepSeekAISummary
from . qcri import FanarLLM, FanarConfig, FanarSummary
from . mistralai import MistralAILLM, MistralAIConfig, MistralAISummary
from . rednote_hilab import RednoteLLM, RednoteConfig, RednoteSummary
from . x_ai import XAILLM, XAIConfig, XAISummary
from . cohere import CohereLLM, CohereConfig,CohereSummary
from . moonshotai import MoonshotAILLM, MoonshotAIConfig, MoonshotAISummary
from . meta_llama import MetaLLM, MetaConfig, MetaSummary
from . microsoft import MicrosoftLLM, MicrosoftConfig, MicrosoftSummary
from . zer01_ai import Zer01AILLM, Zer01AIConfig, Zer01AISummary
from . ai21labs import AI21LLM, AI21Config, AI21Summary
from . allenai import AllenAILLM, AllenAIConfig, AllenAISummary
from . ibm_granite import IBMLLM, IBMConfig, IBMSummary
from .tngtech import TngTechLLM, TngTechConfig, TngTechSummary

MODEL_REGISTRY: Dict[str, Dict[str, type]] = {
    "openai": {
        "LLM_class": OpenAILLM,
        "config_class": OpenAIConfig,
        "summary_class": OpenAISummary,
    },
    "anthropic": {
        "LLM_class": AnthropicLLM,
        "config_class": AnthropicConfig,
        "summary_class": AnthropicSummary,
    },
    "google": {
        "LLM_class": GoogleLLM,
        "config_class": GoogleConfig,
        "summary_class": GoogleSummary
    },
    "deepseek-ai": {
        "LLM_class": DeepSeekAILLM,
        "config_class": DeepSeekAIConfig,
        "summary_class": DeepSeekAISummary
    },
    "fanar": {
        "LLM_class": FanarLLM,
        "config_class": FanarConfig,
        "summary_class": FanarSummary
    },
    "mistralai": {
        "LLM_class": MistralAILLM,
        "config_class": MistralAIConfig,
        "summary_class": MistralAISummary
    },
    "rednote": {
        "LLM_class": RednoteLLM,
        "config_class": RednoteConfig,
        "summary_class": RednoteSummary
    },
    "alibaba": {
        "LLM_class": AlibabaLLM,
        "config_class": AlibabaConfig,
        "summary_class": AlibabaSummary
    },
    "xai": {
        "LLM_class": XAILLM,
        "config_class": XAIConfig,
        "summary_class": XAISummary
    },
    "cohere": {
        "LLM_class": CohereLLM,
        "config_class": CohereConfig,
        "summary_class": CohereSummary
    },
    "moonshotai": {
        "LLM_class": MoonshotAILLM,
        "config_class": MoonshotAIConfig,
        "summary_class": MoonshotAISummary
    },
    "meta": {
        "LLM_class": MetaLLM,
        "config_class": MetaConfig,
        "summary_class": MetaSummary
    },
    "microsoft": {
        "LLM_class": MicrosoftLLM,
        "config_class": MicrosoftConfig,
        "summary_class": MicrosoftSummary
    },
    "01-ai": {
        "LLM_class": Zer01AILLM,
        "config_class": Zer01AIConfig,
        "summary_class": Zer01AISummary
    },
    "ai21": {
        "LLM_class": AI21LLM,
        "config_class": AI21Config,
        "summary_class": AI21Summary
    },
    "allenai": {
        "LLM_class": AllenAILLM,
        "config_class": AllenAIConfig,
        "summary_class": AllenAISummary
    },
    "ibm": {
        "LLM_class": IBMLLM,
        "config_class": IBMConfig,
        "summary_class": IBMSummary
    },
    "tngtech": {
        "LLM_class": TngTechLLM,
        "config_class": TngTechConfig,
        "summary_class": TngTechSummary
    }
}

# Discourage from using `from .LLMs import *` -- Forrest, 2025-07-03
# __all__ = [
#     "AbstractLLM",
#     "MODEL_REGISTRY"
# ]