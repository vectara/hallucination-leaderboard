from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from huggingface_hub import InferenceClient
import re

from src.LLMs.model_registry import register_model

@register_model("deepseek-ai")
class DeepSeekAI(AbstractLLM):
    """
    Class for models from DeepSeekAI

    Attributes:
        client (InferenceClient): client associated with api calls
        min_throttle_time (int): minimum time require per request to avoid
            throttling with huggingface pro
    """
    ds_local = []
    ds1 = ["DeepSeek-R1"]

    def __init__(self, model_name, date_code=""):
        super().__init__(model_name=model_name, company="deepseek-ai", min_throttle_time=4)
        company_model= f"{self.company}/{self.model_name}"
        self.model = self.setup_model_identifier(company_model, date_code)
        self.client = InferenceClient(model=self.model)

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model in self.ds1:
            messages = [{"role": "user", "content":prepared_text}]
            client_package = self.client.chat_completion(messages, temperature=self.temperature)
            raw_summary = client_package.choices[0].message.content
            summary = self.remove_thinking_text(raw_summary)
        return summary

    def setup(self):
        if self.model_name in self.ds_local:
            pass
        else:
            pass

    def teardown(self):
        if self.model_name in self.ds_local:
            pass
        else:
            pass