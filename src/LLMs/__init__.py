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
from . OpenAI import OpenAILLM, OpenAIConfig, OpenAISummary, OpenAIJudgment
from . Anthropic import AnthropicLLM, AnthropicConfig, AnthropicSummary, AnthropicJudgment
from . Google import GoogleLLM, GoogleConfig, GoogleSummary, GoogleJudgment
from . DeepSeekAI import DeepSeekAILLM, DeepSeekAIConfig, DeepSeekAISummary, DeepSeekAIJudgment
from . Fanar import FanarLLM, FanarConfig, FanarSummary, FanarJudgment
from . MistralAI import MistralAILLM, MistralAIConfig, MistralAISummary, MistralAIJudgment
from . Rednote import RednoteLLM, RednoteConfig, RednoteSummary, RednoteJudgment

MODEL_REGISTRY: Dict[str, Dict[str, type]] = {
    "openai": {
        "LLM_class": OpenAILLM,
        "config_class": OpenAIConfig,
        "summary_class": OpenAISummary,
        "judgment_class": OpenAIJudgment
    },
    "anthropic": {
        "LLM_class": AnthropicLLM,
        "config_class": AnthropicConfig,
        "summary_class": AnthropicSummary,
        "judgment_class": AnthropicJudgment
    },
    "google": {
        "LLM_class": GoogleLLM,
        "config_class": GoogleConfig,
        "summary_class": GoogleSummary,
        "judgment_class": GoogleJudgment
    },
    "deepseek-ai": {
        "LLM_class": DeepSeekAILLM,
        "config_class": DeepSeekAIConfig,
        "summary_class": DeepSeekAISummary,
        "judgment_class": DeepSeekAIJudgment
    },
    "fanar": {
        "LLM_class": FanarLLM,
        "config_class": FanarConfig,
        "summary_class": FanarSummary,
        "judgment_class": FanarJudgment
    },
    "mistralai": {
        "LLM_class": MistralAILLM,
        "config_class": MistralAIConfig,
        "summary_class": MistralAISummary,
        "judgment_class": MistralAIJudgment
    },
    "rednote": {
        "LLM_class": RednoteLLM,
        "config_class": RednoteConfig,
        "summary_class": RednoteSummary,
        "judgment_class": RednoteJudgment
    }
}

# Discourage from using `from .LLMs import *` -- Forrest, 2025-07-03
# __all__ = [
#     "AbstractLLM",
#     "MODEL_REGISTRY"
# ]