from src.LLMs.AbstractLLM import AbstractLLM
from openai import OpenAI
import os

from src.LLMs.model_registry import register_model

@register_model("fanar")
class Fanar(AbstractLLM):
    """
    Class for models from Fanar

    Attributes:
        client (OpenAI): client associated with api calls
    """

    def __init__(self, model_name, date_code=None):
        super().__init__(model_name=model_name, company="Fanar")
        api_key = os.getenv("FANAR_API_KEY")
        self.client = OpenAI(
            base_url="https://api.fanar.qa/v1",
            api_key=api_key
        )
        self.model = f"{model_name}"
        if date_code is not None and date_code != "":
            self.model = f"{model_name}-{date_code}"

    def summarize(self, prepared_text: str) -> str:
        chat_package = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content":prepared_text}]
        )
        summary = chat_package.choices[0].message.content
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass
