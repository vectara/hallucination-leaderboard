import os
from google import genai
from google.genai import types
from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from src.LLMs.model_registry import register_model

@register_model("google")
class Google(AbstractLLM):
    """
    Class for models from Google

    Class Attributes:
        g_local (list[str]): models that run locally
        g1 (list[str]): first list of models that follow the same summarize 
            protocol

    Attributes:
        client (str): client associated with api calls
        model (str): google style model name

    """

    local_model_category = []

    # gemini-2.5-pro-preview requieres large output token amount, set to 4096
    model_category1 = ["gemini-2.5-pro-preview"]

    def __init__(self, model_name, date_code, temperature, max_tokens):
        super().__init__(
            model_name, 
            date_code,
            temperature=temperature,
            max_tokens=max_tokens,
            company="google",
            min_throttle_time=9
        )
        api_key = os.getenv("GEMINI_API_KEY")
        self.model = self.get_model_identifier(model_name, date_code)
        if self.model_name not in self.local_model_category:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model_name in self.model_category1 and self.client:
            response = self.client.models.generate_content(
                model = self.model,
                contents=prepared_text,
                config=types.GenerateContentConfig(
                    max_output_tokens = 4096,
                    temperature = self.temperature
                )
            )
            summary = response.text
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
