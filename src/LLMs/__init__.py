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
from . qwen import QwenLLM, QwenConfig, QwenSummary
from . google import GoogleLLM, GoogleConfig, GoogleSummary
from . deepseek_ai import DeepSeekAILLM, DeepSeekAIConfig, DeepSeekAISummary
from . qcri import QCRILLM, QCRIConfig, QCRISummary
from . mistralai import MistralAILLM, MistralAIConfig, MistralAISummary
from . rednote_hilab import RednoteHilabLLM, RednoteHilabConfig, RednoteHilabSummary
from . xai import XAILLM, XAIConfig, XAISummary
from . cohere import CohereLLM, CohereConfig,CohereSummary
from . moonshotai import MoonshotAILLM, MoonshotAIConfig, MoonshotAISummary
from . meta_llama import MetaLlamaLLM, MetaLlamaConfig, MetaLlamaSummary
from . microsoft import MicrosoftLLM, MicrosoftConfig, MicrosoftSummary
from . _01ai import _01AILLM, _01AIConfig, _01AISummary
from . ai21labs import AI21LabsLLM, AI21LabsConfig, AI21LabsSummary
from . allenai import AllenAILLM, AllenAIConfig, AllenAISummary
from . ibm_granite import IBMGraniteLLM, IBMGraniteConfig, IBMGraniteSummary
from . tngtech import TngTechLLM, TngTechConfig, TngTechSummary
from . antgroup_mi import AntGroupMILLM, AntGroupMIConfig, AntGroupMISummary

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
    "qcri": {
        "LLM_class": QCRILLM,
        "config_class": QCRIConfig,
        "summary_class": QCRISummary
    },
    "mistralai": {
        "LLM_class": MistralAILLM,
        "config_class": MistralAIConfig,
        "summary_class": MistralAISummary
    },
    "rednote-hilab": {
        "LLM_class": RednoteHilabLLM,
        "config_class": RednoteHilabConfig,
        "summary_class": RednoteHilabSummary
    },
    "qwen": {
        "LLM_class": QwenLLM,
        "config_class": QwenConfig,
        "summary_class": QwenSummary
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
    "meta-llama": {
        "LLM_class": MetaLlamaLLM,
        "config_class": MetaLlamaConfig,
        "summary_class": MetaLlamaSummary
    },
    "microsoft": {
        "LLM_class": MicrosoftLLM,
        "config_class": MicrosoftConfig,
        "summary_class": MicrosoftSummary
    },
    "01-ai": {
        "LLM_class": _01AILLM,
        "config_class": _01AIConfig,
        "summary_class": _01AISummary
    },
    "ai21labs": {
        "LLM_class": AI21LabsLLM,
        "config_class": AI21LabsConfig,
        "summary_class": AI21LabsSummary
    },
    "allenai": {
        "LLM_class": AllenAILLM,
        "config_class": AllenAIConfig,
        "summary_class": AllenAISummary
    },
    "ibm-granite": {
        "LLM_class": IBMGraniteLLM,
        "config_class": IBMGraniteConfig,
        "summary_class": IBMGraniteSummary
    },
    "tngtech": {
        "LLM_class": TngTechLLM,
        "config_class": TngTechConfig,
        "summary_class": TngTechSummary
    },
    "antgroup-mi": {
        "LLM_class": AntGroupMILLM,
        "config_class": AntGroupMIConfig,
        "summary_class": AntGroupMISummary
    }
}

# Discourage from using `from .LLMs import *` -- Forrest, 2025-07-03
# __all__ = [
#     "AbstractLLM",
#     "MODEL_REGISTRY"
# ]