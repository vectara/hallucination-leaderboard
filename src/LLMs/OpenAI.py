from src.LLMs.AbstractLLM import AbstractLLM
import os
from openai import OpenAI
from src.LLMs.model_registry import register_model

@register_model("openai")
class OpenAi(AbstractLLM):
    """
    Class for models from OpenAI

    Class Attributes:
        gpt_4: list of gpt_4 tier models that follow the same protocol
            for getting a summary

    Attributes:
        client (OpenAI): client associated with api calls
        model (str): exact model name expected by OpenAI
    """

    gpt_4 = ["gpt-4.1"]
    def __init__(self, model_name, date_code=None):
        super().__init__(model_name=model_name, company="openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if date_code:
            self.full_model_name = f"{self.company}/{self.model_name}-{date_code}"
        else:
            self.full_model_name = f"{self.company}/{self.model_name}"
        self.client = OpenAI(api_key=api_key)
        self.model = f"{model_name}"

    def summarize(self, prepared_text: str) -> str:
        summary = None
        if self.model_name in self.gpt_4:
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