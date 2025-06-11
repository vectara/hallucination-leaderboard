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

    open1 = ["gpt-4.1"]
    open2 = ["o3"] # O3 doesnt support temperature of 0.0
    open3 = ["o3-pro"] # o3-pro doesnt support chatting, also doesnt suppor temp
    def __init__(self, model_name, date_code=None):
        super().__init__(model_name=model_name, company="openai")
        api_key = os.getenv("OPENAI_API_KEY")
        self.model = f"{model_name}"
        if date_code is not None and date_code != "":
            self.model = f"{model_name}-{date_code}"
        self.client = OpenAI(api_key=api_key)

    def summarize(self, prepared_text: str) -> str:
        summary = None
        if self.model_name in self.open1:
            chat_package = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content":prepared_text}]
            )
            summary = chat_package.choices[0].message.content
        elif self.model_name in self.open2:
            chat_package = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_completion_tokens=self.max_tokens
            )
            summary = chat_package.choices[0].message.content
        elif self.model_name in self.open3:
            chat_package = self.client.responses.create(
                model=self.model,
                input=prepared_text,
                max_output_tokens=self.max_tokens
            )
            summary = chat_package.output_text

        return summary

    def setup(self):
        pass

    def teardown(self):
        pass