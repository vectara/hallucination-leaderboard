# import os
# import importlib

"""
Import all LLM classes
"""

# LLM_dir = os.path.dirname(__file__)

# for file in os.listdir(LLM_dir):
#     if file.endswith(".py") and file != "__init__.py":
#         file_name = file[:-3]
#         importlib.import_module(f"src.LLMs.{file_name}")


from AbstractLLM import AbstractLLM

# TODO: Move below to either `main.py` or `config.py`
from Anthropic import Anthropic
from DeepSeekAI import DeepSeekAI
from MistralAI import MistralAI
from Fanar import Fanar
from Google import Google
from OpenAI import OpenAI
from Rednote import Rednote

MODEL_REGISTRY = { # keys as company names on HuggingFace
    'anthropic': Anthropic,
    'deepseek-ai': DeepSeekAI,
    'mistralai': MistralAI,
    'fanar': Fanar,
    'openai': OpenAI,
    'rednote': Rednote,
    'google': Google
}