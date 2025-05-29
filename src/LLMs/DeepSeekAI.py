from src.LLMs.AbstractLLM import AbstractLLM
from transformers import pipeline
import torch
import gc
import os

class DeepSeekAI(AbstractLLM):

    def __init__(self, model_name, data_code=None):
        super().__init__(model_name=model_name, company="deepseek-ai")
        self.full_model_name = f"{self.company}/{self.model_name}"
        self.model = None

    def summarize(self, prepared_text: str) -> str:
        summary = self.model(prepared_text, temperature=0)[0]['generated_text']
        return summary

    def setup(self):
        self.model = pipeline("text-generation", model=self.full_model_name)

    def teardown(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()