from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
import os
from openai import OpenAI
from src.LLMs.model_registry import register_model
from src.data_struct.config_model import ExecutionMode

COMPANY ="openai"
@register_model(COMPANY)
class OpenAi(AbstractLLM):
    """
    Class for models from OpenAI

    Attributes:
        client (OpenAI): client associated with api calls
        model (str): OpenAI style model name
    """

    local_models = []
    client_models = ["gpt-4.1", "o3", "o3-pro"]

    model_category1 = ["gpt-4.1"]

    # o3 doesn't support adjusting the temperature
    model_category2 = ["o3"]

    # o3 doesn't support chatting protocol and doesn't support adjusting temperature
    model_category3 = ["o3-pro"]

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
            chat_package = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content":prepared_text}]
            )
            summary = chat_package.choices[0].message.content
        elif self.client and self.model_name in self.model_category2:
            chat_package = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_completion_tokens=self.max_tokens
            )
            summary = chat_package.choices[0].message.content
        elif self.client and self.model_name in self.model_category3:
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
        if self.valid_client_model():
            api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
            self.client = OpenAI(api_key=api_key)
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
