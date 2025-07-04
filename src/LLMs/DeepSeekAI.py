from src.LLMs.AbstractLLM import AbstractLLM, MODEL_REGISTRY
from huggingface_hub import InferenceClient
from src.config_model import ExecutionMode, InteractionMode
from src.LLMs.AbstractLLM import SummaryError, ModelInstantiationError

COMPANY = "deepseek-ai"
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
        company_model= f"{self.company}/{self.model_name}"
        self.model = self.get_model_identifier(company_model, date_code)

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client_is_defined():
            if self.model_name in self.model_category1:
                messages = [{"role": "user", "content":prepared_text}]
                client_package = self.client.chat_completion(messages, temperature=self.temperature)
                summary = client_package.choices[0].message.content
            else:
                raise ModelInstantiationError.NOT_REGISTERED(self.model_name, self.company, self.execution_mode)
        elif self.local_model_is_defined():
            pass
        else:
            raise ModelInstantiationError.MISSING_SETUP(self.__class__.__name__)
        return summary

    def setup(self):
        if self.valid_client_model():
            self.client = InferenceClient(model=self.model)
        elif self.valid_local_model():
            pass

    def teardown(self):
        if self.client_is_defined():
            self.close_client()
        elif self.local_model_is_defined():
            self.default_local_model_teardown()

    def close_client(self):
        pass

MODEL_REGISTRY[COMPANY] = DeepSeekAI
