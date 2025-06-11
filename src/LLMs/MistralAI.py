from src.LLMs.AbstractLLM import AbstractLLM
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
    m1 = ["magistral-medium"]

    def __init__(self, model_name, date_code=None):
        super().__init__(model_name=model_name, company="mistralai")
        api_key = os.getenv("MISTRALAI_API_KEY")
        self.model = f"{model_name}"
        if date_code is not None and date_code != "":
            self.model = f"{model_name}-{date_code}"
        self.client = Mistral(api_key=api_key)

    def summarize(self, prepared_text: str) -> str:
        summary = None
        if self.model_name in self.m1:
            chat_package = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            raw_summary = chat_package.choices[0].message.content
            summary = re.sub(r"<think>.*?</think>\n*", "", raw_summary, flags=re.DOTALL)
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass