import os
import importlib

"""
Import all LLM classes
"""

LLM_dir = os.path.dirname(__file__)

for file in os.listdir(LLM_dir):
    if file.endswith(".py") and file != "__init__.py":
        file_name = file[:-3]
        importlib.import_module(f"src.LLMs.{file_name}")