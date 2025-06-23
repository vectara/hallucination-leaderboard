from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from src.LLMs.model_registry import register_model
from mistralai import Mistral
import os
import re


@register_model("mistralai")
class MistralAI(AbstractLLM):
    """
    Class for models from MistralAI

    Class Attributes:
        mist_local (list[str]): models that run locally
        mist1 (list[str]) first list of models that follow the same summarize
            prototocol

    Attributes:
        client (str): client associated with api calls
        model (str): MistralAI Style model name
    """

    local_model_category = []

    model_category1 = ["magistral-medium"]

    def __init__(self, model_name, date_code):
        super().__init__(
            model_name,
            date_code,
            company="mistralai"
        )
        api_key = os.getenv("MISTRALAI_API_KEY")
        self.model = self.get_model_identifier(model_name, date_code)
        if self.model_name not in self.mist_local:
            self.client = Mistral(api_key=api_key)
        else:
            self.client = None

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model_name in self.model_category1 and self.client:
            chat_package = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_tokens=4096,
                temperature=self.temperature
            )
            summary = chat_package.choices[0].message.content
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
