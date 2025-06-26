from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from huggingface_hub import InferenceClient
import re

from src.LLMs.model_registry import register_model
from src.data_struct.config_model import ExecutionMode

COMPANY = "deepseek-ai"
@register_model(COMPANY)
class DeepSeekAI(AbstractLLM):
    """
    Class for models from DeepSeekAI

    Attributes:
        client (InferenceClient): client associated with api calls
        self.model (str): DeepSeekAI style model name
    """

    local_models = []
    client_models = ["DeepSeek-R1"]

    model_category1 = ["DeepSeek-R1"]

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
        company_model= f"{self.company}/{self.model_name}"
        self.model = self.get_model_identifier(company_model, date_code)

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.client and self.model_name in self.model_category1:
            messages = [{"role": "user", "content":prepared_text}]
            client_package = self.client.chat_completion(messages, temperature=self.temperature)
            summary = client_package.choices[0].message.content
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return summary

    def setup(self):
        if self.valid_client_model():
            self.client = InferenceClient(model=self.model)
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
