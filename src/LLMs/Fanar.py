from src.LLMs.AbstractLLM import AbstractLLM
from openai import OpenAI
import os

from src.LLMs.model_registry import register_model

@register_model("fanar")
class Fanar(AbstractLLM):

    def __init__(self, model_name, data_code=None):
        super().__init__(model_name=model_name, company="Fanar")
        self.api_key = os.getenv("FANAR_API_KEY")
        self.client = OpenAI(
            base_url="https://api.fanar.qa/v1",
            api_key=self.api_key
        )

    def summarize(self, prepared_text: str) -> str:
        chat_package = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[{"role": "user", "content":prepared_text}]
        )
        summary = chat_package.choices[0].message.content
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass
