from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from openai import OpenAI
import os

from src.LLMs.model_registry import register_model

@register_model("fanar")
class Fanar(AbstractLLM):
    """
    Class for models from Fanar

    Class Attributes:
        fan_local (list[str]): list of local models
        fan1 (list[str]): first list of models that follow similar summarize
            protocol

    Attributes:
        client (OpenAI): client associated with api calls
        model (str): Fanar style model name
    """

    fan_local = []

    fan = ["Fanar"]

    def __init__(self, model_name, date_code=""):
        super().__init__(model_name=model_name, company="Fanar")
        api_key = os.getenv("FANAR_API_KEY")
        self.model = self.get_model_identifier(model_name, date_code)
        if self.model_name not in self.fan_local:
            self.client = OpenAI(
                base_url="https://api.fanar.qa/v1",
                api_key=api_key
            )
        else:
            self.client = None

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model in self.fan and self.client:
            chat_package = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content":prepared_text}]
            )
            summary = chat_package.choices[0].message.content
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return summary

    def setup(self):
        if self.model_name in self.fan_local:
            pass
        else:
            pass

    def teardown(self):
        if self.model_name in self.fan_local:
            pass
        else:
            pass
