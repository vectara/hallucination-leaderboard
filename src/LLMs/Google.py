import os
from google import genai
from google.genai import types
from src.LLMs.AbstractLLM import AbstractLLM, MODEL_REGISTRY
from src.config_model import ExecutionMode, InteractionMode
from src.LLMs.AbstractLLM import SummaryError, ModelInstantiationError

COMPANY = "google"
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
            interaction_mode: InteractionMode,
            date_code: str,
            temperature: float,
            max_tokens: int,
            thinking_tokens: int,
            min_throttle_time: float
    ):
        super().__init__(
            model_name,
            execution_mode,
            interaction_mode,
            date_code,
            temperature,
            max_tokens,
            thinking_tokens,
            min_throttle_time,
            company=COMPANY
        )
        self.model = self.get_model_identifier(model_name, date_code)

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client_is_defined():
            if self.model_name in self.model_category1:
                response = self.client.models.generate_content(
                    model = self.model,
                    contents=prepared_text,
                    config=types.GenerateContentConfig(
                        max_output_tokens = 4096,
                        temperature = self.temperature
                    )
                )
                summary = response.text
            elif self.model_name in self.model_category2:
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
                raise ModelInstantiationError.NOT_REGISTERED(self.model_name, self.company, self.execution_mode)
        elif self.local_model_is_defined():
            pass
        else:
            raise ModelInstantiationError.MISSING_SETUP(self.__class__.__name__)
        return summary

    def setup(self):
        if self.valid_client_model():
            api_key = os.getenv(f"{COMPANY.upper()}_GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
        elif self.valid_local_model():
            pass

    def teardown(self):
        if self.client_is_defined():
            self.close_client()
        elif self.local_model_is_defined():
            self.default_local_model_teardown()

    def close_client(self):
        pass

MODEL_REGISTRY[COMPANY] = Google
