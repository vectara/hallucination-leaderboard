from src.LLMs.AbstractLLM import AbstractLLM, MODEL_REGISTRY
from openai import OpenAI
import os
from src.config_model import ExecutionMode, InteractionMode
from src.LLMs.AbstractLLM import SummaryError, ModelInstantiationError

COMPANY = "fanar"
class Fanar(AbstractLLM):
    """
    Class for models from Fanar

    Attributes:
        client (OpenAI): client associated with api calls
        model (str): Fanar style model name
    """

    local_models = []
    client_models = ["Fanar"]

    model_category1 = ["Fanar"]

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
                chat_package = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content":prepared_text}]
                )
                summary = chat_package.choices[0].message.content
            else:
                raise ModelInstantiationError.NOT_REGISTERED(self.model_name, self.company, self.execution_mode)
        elif self.local_model_is_defined():
            pass
        else:
            raise ModelInstantiationError.MISSING_SETUP(self.__class__.__name__)
        return summary

    def setup(self):
        if self.valid_client_model():
            api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
            self.client = OpenAI(
                base_url="https://api.fanar.qa/v1",
                api_key=api_key
            )
        elif self.valid_local_model():
            pass

    def teardown(self):
        if self.client_is_defined():
            self.close_client()
        elif self.local_model_is_defined():
            self.default_local_model_teardown()

    def close_client(self):
        pass

MODEL_REGISTRY[COMPANY] = Fanar
