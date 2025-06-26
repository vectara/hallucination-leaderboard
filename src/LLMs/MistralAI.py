from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from src.LLMs.model_registry import register_model
from mistralai import Mistral
import os
import re
from src.data_struct.config_model import ExecutionMode

COMPANY = "mistralai"
@register_model(COMPANY)
class MistralAI(AbstractLLM):
    """
    Class for models from MistralAI

    Attributes:
        client (str): client associated with api calls
        model (str): MistralAI Style model name
    """

    local_models = []
    client_models = ["magistral-medium"]

    model_category1 = ["magistral-medium"] # Doesn't look like magistral can disable thinking

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
            chat_package = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            summary = chat_package.choices[0].message.content
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return summary

    def setup(self):
        if self.valid_client_model():
            api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
            self.client = Mistral(api_key=api_key)
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
