from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from huggingface_hub import InferenceClient
import re

from src.LLMs.model_registry import register_model

COMPANY = "deepseek-ai"
@register_model(COMPANY)
class DeepSeekAI(AbstractLLM):
    """
    Class for models from DeepSeekAI

    Attributes:
        client (InferenceClient): client associated with api calls
        self.model (str): DeepSeekAI style model name
    """

    local_model = []
    client_model = ["DeepSeek-R1"]

    model_category1 = ["DeepSeek-R1"]

    def __init__(
            self,
            model_name: str,
            date_code: str,
            temperature: float,
            max_tokens: int,
            thinking_tokens: int,
            min_throttle_time: float
    ):
        super().__init__(
            model_name,
            date_code,
            temperature,
            max_tokens,
            thinking_tokens,
            min_throttle_time,
            company=COMPANY
        )
        company_model= f"{self.company}/{self.model_name}"
        self.model = self.get_model_identifier(company_model, date_code)
        if self.model_name in self.client_model:
            self.client = InferenceClient(model=self.model)
        else:
            self.client = None

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.valid_client_model(self.model, self.model_category1):
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