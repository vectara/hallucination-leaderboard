from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
import os
import anthropic
from src.LLMs.model_registry import register_model

COMPANY = "anthropic"
@register_model(COMPANY)
class Anthropic(AbstractLLM):
    """
    Class for models from Anthropic

    Attributes:
        client (Client): client associated with api calls with anthropic
        model (str): anthropic style model name
    """

    local_models = []
    client_models = ["claude-opus-4", "claude-sonnet-4"]

    model_category1 = ["claude-opus-4", "claude-sonnet-4"]

    def __init__(
            self,
            model_name: str,
            execution_mode: str,
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
        if self.client and self.model in self.model_category1:
            chat_package = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            summary = chat_package.content[0].text
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return summary

    def setup(self):
        if self.valid_client_model():
            api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
            self.client = anthropic.Client(api_key=api_key)
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
