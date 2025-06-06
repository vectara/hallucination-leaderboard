import os
from google import genai
from google.genai import types
from src.LLMs.AbstractLLM import AbstractLLM
from src.LLMs.model_registry import register_model

@register_model("google")
class Google(AbstractLLM):
    """

    Class Attributes:

    Attributes:
    """



    def __init__(self, model_name, date_code=None):
        super().__init__(model_name=model_name, company="google", min_throttle_time=0.5)
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = f"{model_name}"
        if date_code is not None and date_code != "":
            self.model = f"{model_name}-{date_code}"

    def summarize(self, prepared_text: str) -> str:
        summary = None
        response = self.client.models.generate_content(
            model = self.model,
            contents=prepared_text,
            config=types.GenerateContentConfig(
                max_output_tokens = self.max_tokens,
                temperature = 0.10
            )
        )
        summary = response.text
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass