"""LLM provider implementations for hallucination evaluation.

This package contains implementations for various LLM providers used in the
hallucination leaderboard evaluation pipeline. All providers implement a unified
interface through the AbstractLLM base class, enabling consistent summarization
across different models.

Each provider module exports three components:
    - LLM class: The main class implementing AbstractLLM for inference.
    - Config class: Pydantic model for provider-specific configuration.
    - Summary class: Pydantic model for structured summary output.

Key AbstractLLM Methods:
    setup: Initialize the LLM connection at the start of a context block.
    teardown: Clean up resources at the end of a context block.
    summarize: Generate a summary of the provided text.

Example:
    >>> from src.LLMs import MODEL_REGISTRY
    >>> provider_info = MODEL_REGISTRY["openai"]
    >>> config = provider_info["config_class"](model_name="gpt-4o")
    >>> with provider_info["LLM_class"](config) as llm:
    ...     result = llm.summarize("Long article text here...")

Attributes:
    MODEL_REGISTRY: Dictionary mapping provider names to their LLM, config,
        and summary classes for dynamic instantiation.
"""

from typing import Dict, Any

from . AbstractLLM import AbstractLLM

from . _01ai import _01AILLM, _01AIConfig, _01AISummary
from . ai21labs import AI21LabsLLM, AI21LabsConfig, AI21LabsSummary
from . allenai import AllenAILLM, AllenAIConfig, AllenAISummary
from . amazon import AmazonLLM, AmazonConfig, AmazonSummary
from . antgroup_mi import AntGroupMILLM, AntGroupMIConfig, AntGroupMISummary
from . anthropic import AnthropicLLM, AnthropicConfig, AnthropicSummary
from . arcee_ai import ArceeAILLM, ArceeAIConfig, ArceeAISummary
from . apple import AppleLLM, AppleConfig, AppleSummary
from . baidu import BaiduLLM, BaiduConfig, BaiduSummary
from . cohere import CohereLLM, CohereConfig,CohereSummary
from . databricks import DatabricksLLM, DatabricksConfig, DatabricksSummary
from . deepseek_ai import DeepSeekAILLM, DeepSeekAIConfig, DeepSeekAISummary
from . google import GoogleLLM, GoogleConfig, GoogleSummary
from . ibm_granite import IBMGraniteLLM, IBMGraniteConfig, IBMGraniteSummary
from . intel import IntelLLM, IntelConfig, IntelSummary
from . internlm import InternLmLLM, InternLmConfig, InternLmSummary
from . meta_llama import MetaLlamaLLM, MetaLlamaConfig, MetaLlamaSummary
from . microsoft import MicrosoftLLM, MicrosoftConfig, MicrosoftSummary
from . minimaxai import MiniMaxAILLM, MiniMaxAIConfig, MiniMaxAISummary
from . mistralai import MistralAILLM, MistralAIConfig, MistralAISummary
from . moonshotai import MoonshotAILLM, MoonshotAIConfig, MoonshotAISummary
from . nvidia import NvidiaLLM, NvidiaConfig, NvidiaSummary
from . openai import OpenAILLM, OpenAIConfig, OpenAISummary
from . primeintellect import PrimeIntellectLLM, PrimeIntellectConfig, PrimeIntellectSummary
from . qcri import QCRILLM, QCRIConfig, QCRISummary
from . qwen import QwenLLM, QwenConfig, QwenSummary
from . rednote_hilab import RednoteHilabLLM, RednoteHilabConfig, RednoteHilabSummary
from . snowflake import SnowflakeLLM, SnowflakeConfig, SnowflakeSummary
from . tiiuae import TiiuaeLLM, TiiuaeConfig, TiiuaeSummary
from . tngtech import TngTechLLM, TngTechConfig, TngTechSummary
from . vectara import VectaraLLM, VectaraConfig, VectaraSummary
from . xai import XAILLM, XAIConfig, XAISummary
from . zai_org import ZhipuAILLM, ZhipuAIConfig, ZhipuAISummary


# MODEL_REGISTRY: Central registry for all supported LLM providers.
#
# Maps provider identifier strings to dictionaries containing:
#     - LLM_class: The AbstractLLM subclass for model inference.
#     - config_class: The Pydantic config model for provider settings.
#     - summary_class: The Pydantic model for structured summary output.
#
# Provider keys should match the organization/provider naming convention
# used in model identifiers (e.g., "openai", "anthropic", "meta-llama").
#
# To add a new provider:
#     1. Create a new module in src/LLMs/ with LLM, Config, and Summary classes.
#     2. Import the classes at the top of this file.
#     3. Add an entry to MODEL_REGISTRY with the provider key.
MODEL_REGISTRY: Dict[str, Dict[str, type]] = {
    "01-ai": {
        "LLM_class": _01AILLM,
        "config_class": _01AIConfig,
        "summary_class": _01AISummary
    },
    "antgroup": {
        "LLM_class": AntGroupMILLM,
        "config_class": AntGroupMIConfig,
        "summary_class": AntGroupMISummary
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
    "amazon": {
        "LLM_class": AmazonLLM,
        "config_class": AmazonConfig,
        "summary_class": AmazonSummary,
    },
    "anthropic": {
        "LLM_class": AnthropicLLM,
        "config_class": AnthropicConfig,
        "summary_class": AnthropicSummary,
    },
    "arcee-ai": {
        "LLM_class": ArceeAILLM,
        "config_class": ArceeAIConfig,
        "summary_class": ArceeAISummary,
    },
    "apple": {
        "LLM_class": AppleLLM,
        "config_class": AppleConfig,
        "summary_class": AppleSummary,
    },
    "baidu": {
        "LLM_class": BaiduLLM,
        "config_class": BaiduConfig,
        "summary_class": BaiduSummary,
    },
    "CohereLabs": {
        "LLM_class": CohereLLM,
        "config_class": CohereConfig,
        "summary_class": CohereSummary
    },
    "databricks": {
        "LLM_class": DatabricksLLM,
        "config_class": DatabricksConfig,
        "summary_class": DatabricksSummary,
    },
    "deepseek-ai": {
        "LLM_class": DeepSeekAILLM,
        "config_class": DeepSeekAIConfig,
        "summary_class": DeepSeekAISummary
    },
    "google": {
        "LLM_class": GoogleLLM,
        "config_class": GoogleConfig,
        "summary_class": GoogleSummary
    },
    "ibm-granite": {
        "LLM_class": IBMGraniteLLM,
        "config_class": IBMGraniteConfig,
        "summary_class": IBMGraniteSummary
    },
    "Intel": {
        "LLM_class": IntelLLM,
        "config_class": IntelConfig,
        "summary_class": IntelSummary
    },
    "internlm": {
        "LLM_class": InternLmLLM,
        "config_class": InternLmConfig,
        "summary_class": InternLmSummary
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
    "MiniMaxAI": {
        "LLM_class": MiniMaxAILLM,
        "config_class": MiniMaxAIConfig,
        "summary_class": MiniMaxAISummary
    },
    "mistralai": {
        "LLM_class": MistralAILLM,
        "config_class": MistralAIConfig,
        "summary_class": MistralAISummary
    },
    "moonshotai": {
        "LLM_class": MoonshotAILLM,
        "config_class": MoonshotAIConfig,
        "summary_class": MoonshotAISummary
    },
    "nvidia": {
        "LLM_class": NvidiaLLM,
        "config_class": NvidiaConfig,
        "summary_class": NvidiaSummary
    },
    "openai": {
        "LLM_class": OpenAILLM,
        "config_class": OpenAIConfig,
        "summary_class": OpenAISummary,
    },
    "PrimeIntellect": {
        "LLM_class": PrimeIntellectLLM,
        "config_class": PrimeIntellectConfig,
        "summary_class": PrimeIntellectSummary,
    },
    "qcri": {
        "LLM_class": QCRILLM,
        "config_class": QCRIConfig,
        "summary_class": QCRISummary
    },
    "qwen": {
        "LLM_class": QwenLLM,
        "config_class": QwenConfig,
        "summary_class": QwenSummary
    },
    "rednote-hilab": {
        "LLM_class": RednoteHilabLLM,
        "config_class": RednoteHilabConfig,
        "summary_class": RednoteHilabSummary
    },
    "snowflake": {
        "LLM_class": SnowflakeLLM,
        "config_class": SnowflakeConfig,
        "summary_class": SnowflakeSummary
    },
    "tiiuae": {
        "LLM_class": TiiuaeLLM,
        "config_class": TiiuaeConfig,
        "summary_class": TiiuaeSummary
    },
    "tngtech": {
        "LLM_class": TngTechLLM,
        "config_class": TngTechConfig,
        "summary_class": TngTechSummary
    },
    "vectara": {
        "LLM_class": VectaraLLM,
        "config_class": VectaraConfig,
        "summary_class": VectaraSummary
    },
    "xai-org": {
        "LLM_class": XAILLM,
        "config_class": XAIConfig,
        "summary_class": XAISummary
    },
    "zai-org": {
        "LLM_class": ZhipuAILLM,
        "config_class": ZhipuAIConfig,
        "summary_class": ZhipuAISummary
    }
}

# Note: __all__ is intentionally not defined to discourage wildcard imports.
# Use explicit imports: `from src.LLMs import AbstractLLM, MODEL_REGISTRY`