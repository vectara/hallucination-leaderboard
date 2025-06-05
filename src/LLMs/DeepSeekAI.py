from src.LLMs.AbstractLLM import AbstractLLM
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
    def __init__(self, model_name, date_code=None):
        super().__init__(model_name=model_name, company="deepseek-ai", min_throttle_time=4)
        self.full_model_name = f"{self.company}/{self.model_name}"
        self.client = InferenceClient(model=self.full_model_name)

    def summarize(self, prepared_text: str) -> str:
        messages = [{"role": "user", "content":prepared_text}]
        client_package = self.client.chat_completion(messages, temperature=self.temperature)
        raw_summary = client_package.choices[0].message.content
        summary = re.sub(r'<think>.*?</think>\s*', '', raw_summary, flags=re.DOTALL)
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass