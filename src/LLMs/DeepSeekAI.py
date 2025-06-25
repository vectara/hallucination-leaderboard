from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from huggingface_hub import InferenceClient
import re

from src.LLMs.model_registry import register_model

@register_model("deepseek-ai")
class DeepSeekAI(AbstractLLM):
    """
    Class for models from DeepSeekAI

    Class Attributes:
        ds_local (list[str]): models that run locally
        ds1 (list[str]): first list of models that follow similar summarize
            protocol

    Attributes:
        client (InferenceClient): client associated with api calls
        self.model (str): DeepSeekAI style model name
    """

    local_model_category = []

    model_category1 = ["DeepSeek-R1"]

    def __init__(self, model_name, date_code, temperature, max_tokens):
        super().__init__(
            model_name,
            date_code,
            temperature=temperature,
            max_tokens=max_tokens,
            company="deepseek-ai",
            min_throttle_time=4
        )
        company_model= f"{self.company}/{self.model_name}"
        self.model = self.get_model_identifier(company_model, date_code)
        if self.model_name not in self.local_model_category:
            self.client = InferenceClient(model=self.model)
        else:
            self.client = None

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model in self.model_category1 and self.client:
            messages = [{"role": "user", "content":prepared_text}]
            client_package = self.client.chat_completion(messages, temperature=self.temperature)
            summary = client_package.choices[0].message.content
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return summary

    def setup(self):
        if self.model_name in self.local_model_category:
            pass
        else:
            pass

    def teardown(self):
        if self.model_name in self.local_model_category:
            pass
        else:
            pass