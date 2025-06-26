import os
from google import genai
from google.genai import types
from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from src.LLMs.model_registry import register_model
from src.data_struct.config_model import ExecutionMode

COMPANY = "google"
@register_model(COMPANY)
class Google(AbstractLLM):
    """
    Class for models from Google

    Attributes:
        client (str): client associated with api calls
        model (str): google style model name

    """

    local_models = []
    client_models = ["gemini-2.5-pro-preview", "gemini-2.5-pro"]

    # gemini-2.5-pro-preview requieres large output token amount, set to 4096
    # 9 throttle time
    model_category1 = ["gemini-2.5-pro-preview"]
    
    model_category2 = ["gemini-2.5-pro"]

    def __init__(
            self,
            model_name: str,
            execution_mode: ExecutionMode,
            date_code: str,
            temperature: float,
            max_tokens: int,
            thinking_tokens: int,
            min_throttle_time: float
    ):
        super().__init__(
            model_name,
            execution_mode,
            date_code,
            temperature,
            max_tokens,
            thinking_tokens,
            min_throttle_time,
            company=COMPANY
        )
        self.model = self.get_model_identifier(model_name, date_code)

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.client and self.model_name in self.model_category1:
            response = self.client.models.generate_content(
                model = self.model,
                contents=prepared_text,
                config=types.GenerateContentConfig(
                    max_output_tokens = 4096,
                    temperature = self.temperature
                )
            )
            summary = response.text
        elif self.client and self.model_name in self.model_category2:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prepared_text,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_tokens)
                ),
            )
            summary = response.text
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return summary

    def setup(self):
        if self.valid_client_model():
            api_key = os.getenv(f"{COMPANY.upper()}_GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
        elif self.valid_local_model():
            pass
        else:
            pass

    def teardown(self):
        if self.valid_client_model():
            pass
        elif self.valid_local_model():
            pass
        else:
            pass
