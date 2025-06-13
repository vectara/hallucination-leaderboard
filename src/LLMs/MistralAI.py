from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from src.LLMs.model_registry import register_model
from mistralai import Mistral
import os
import re


@register_model("mistralai")
class MistralAI(AbstractLLM):
    #TODO: Doc
    """
    Class for models from MistralAI
    
    """
    mist_local = []
    mist1 = ["magistral-medium"]

    def __init__(self, model_name, date_code=""):
        super().__init__(model_name=model_name, company="mistralai")
        api_key = os.getenv("MISTRALAI_API_KEY")
        self.model = self.get_model_identifier(model_name, date_code)
        self.client = Mistral(api_key=api_key)

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model_name in self.mist1:
            chat_package = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_tokens=2048,
                temperature=self.temperature
            )
            summary = chat_package.choices[0].message.content
        return summary

    def setup(self):
        if self.model_name in self.mist_local:
            pass
        else:
            pass

    def teardown(self):
        if self.model_name in self.mist_local:
            pass
        else:
            pass
