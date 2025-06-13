from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
import os
from openai import OpenAI
from src.LLMs.model_registry import register_model

@register_model("openai")
class OpenAi(AbstractLLM):
    """
    Class for models from OpenAI

    Class Attributes:
        open_local (list[str]): models that run locally
        open1 (list[str]): first list of models that follow the same summarize
            protocol
        open2 (list[str]): 2nd list of models that follow the same summarize
            protocol
        open3 (list[str]): 3rd list of models that follow the same summarize
            protocol

    Attributes:
        client (OpenAI): client associated with api calls
        model (str): OpenAI style model name
    """

    open_local = []

    open1 = ["gpt-4.1"]

    # o3 doesn't support adjusting the temperature
    open2 = ["o3"]

    # o3 doesn't support chatting protocol and doesn't support adjusting temperature
    open3 = ["o3-pro"]

    def __init__(self, model_name, date_code=""):
        super().__init__(model_name=model_name, company="openai")
        api_key = os.getenv("OPENAI_API_KEY")
        self.model = self.get_model_identifier(model_name, date_code)
        self.client = OpenAI(api_key=api_key)

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model_name in self.open1 and self.client:
            chat_package = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content":prepared_text}]
            )
            summary = chat_package.choices[0].message.content
        elif self.model_name in self.open2 and self.client:
            chat_package = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_completion_tokens=self.max_tokens
            )
            summary = chat_package.choices[0].message.content
        elif self.model_name in self.open3 and self.client:
            chat_package = self.client.responses.create(
                model=self.model,
                input=prepared_text,
                max_output_tokens=self.max_tokens
            )
            summary = chat_package.output_text
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return summary

    def setup(self):
        if self.model_name in self.open_local:
            pass
        else:
            pass

    def teardown(self):
        if self.model_name in self.open_local:
            pass
        else:
            pass
