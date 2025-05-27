from src.LLMs.AbstractLLM import AbstractLLM
from openai import OpenAI
import os

class GPTd4p1(AbstractLLM):

    def __init__(self):
        super().__init__(name="GPT-4.1", company="OpenAI")
        self.api_key = os.getenv("OPENAI_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model_type = "gpt-4.1"

    def summarize(self, prepared_text: str) -> str:
        chat_package = self.client.chat.completions.create(
            model=self.model_type,
            messages=[{"role": "user", "content":prepared_text}]
        )
        summary = chat_package.choices[0].message.content
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass
