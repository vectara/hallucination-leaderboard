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


from . AbstractLLM import AbstractLLM, BasicLLMConfig
from .. data_model import BasicSummary

# All vendor classes are called _{Vendor}_Class below to avoid name conflicts. -- Forrest, 2025-07-03
from . OpenAI import OpenAI as _OpenAI_Class, OpenAILLMConfig, OpenAISummary
from . Anthropic import Anthropic as _Anthropic_Class, AnthropicConfig, AnthropicSummary
from . Google import Google as _Google_Class, GoogleLLMConfig, GoogleSummary
from . DeepSeekAI import DeepSeekAI as _DeepSeekAI_Class, DeepSeekAILLMConfig, DeepSeekAISummary
from . Fanar import Fanar as _Fanar_Class, FanarLLMConfig, FanarSummary
from . MistralAI import MistralAI as _MistralAI_Class, MistralAILLMConfig, MistralAISummary
from . Rednote import Rednote as _Rednote_Class, RednoteLLMConfig, RednoteSummary

MODEL_REGISTRY: Dict[str, Dict[str, AbstractLLM|BasicLLMConfig|BasicSummary]] = {
    "openai": {
        "LLM_class": _OpenAI_Class,
        "config_class": OpenAILLMConfig,
        "summary_class": OpenAISummary
    },
    "anthropic": {
        "LLM_class": _Anthropic_Class,
        "config_class": AnthropicConfig,
        "summary_class": AnthropicSummary
    },
    "google": {
        "LLM_class": _Google_Class,
        "config_class": GoogleLLMConfig,
        "summary_class": GoogleSummary
    },
    "deepseekai": {
        "LLM_class": _DeepSeekAI_Class,
        "config_class": DeepSeekAILLMConfig,
        "summary_class": DeepSeekAISummary
    },
    "fanar": {
        "LLM_class": _Fanar_Class,
        "config_class": FanarLLMConfig,
        "summary_class": FanarSummary
    },
    "mistralai": {
        "LLM_class": _MistralAI_Class,
        "config_class": MistralAILLMConfig,
        "summary_class": MistralAISummary
    },
    "rednote": {
        "LLM_class": _Rednote_Class,
        "config_class": RednoteLLMConfig,
        "summary_class": RednoteSummary
    }
}

# Discourage from using `from .LLMs import *` -- Forrest, 2025-07-03
# __all__ = [
#     "AbstractLLM",
#     "MODEL_REGISTRY"
# ]