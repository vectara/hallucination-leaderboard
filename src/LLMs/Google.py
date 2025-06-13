import os
from google import genai
from google.genai import types
from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from src.LLMs.model_registry import register_model

@register_model("google")
class Google(AbstractLLM):
    """

    Class Attributes:

    Attributes:
    """
    g_local = []
    g1 = ["gemini-2.5-pro-preview"] #low token size makes this model fail to work


    def __init__(self, model_name, date_code=""):
        super().__init__(model_name=model_name, company="google", min_throttle_time=9)
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = self.get_model_identifier(model_name, date_code)

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model_name in self.g1:
            response = self.client.models.generate_content(
                model = self.model,
                contents=prepared_text,
                config=types.GenerateContentConfig(
                    max_output_tokens = 4096,
                    temperature = self.temperature
                )
            )
            summary = response.text
        return summary

    def setup(self):
        if self.model_name in self.g_local:
            pass
        else:
            pass

    def teardown(self):
        if self.model_name in self.g_local:
            pass
        else:
            pass
