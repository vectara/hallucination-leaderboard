from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
import os
import anthropic
from src.LLMs.model_registry import register_model

@register_model("anthropic")
class Anthropic(AbstractLLM):
    """
    Class for models from Anthropic

    Class Attributes:
        anth_local (list[str]): models that run locally
        anth1 (list[str]): first list of models that follow a similar summarize
            protocol

    Attributes:
        client (Client): client associated with api calls with anthropic
        model (str): anthropic style model name
    """

    local_model_category = []

    model_category1 = ["claude-opus-4", "claude-sonnet-4"]

    def __init__(self, model_name, date_code, temperature, max_tokens):
        super().__init__(
            model_name,
            date_code,
            temperature=temperature,
            max_tokens=max_tokens,
            company="anthropic"
        )
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = self.get_model_identifier(model_name, date_code)
        if self.model_name not in self.local_model_category:
            self.client = anthropic.Client(api_key=api_key)
        else:
            self.client = None

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.model_name in self.model_category1 and self.client:
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
        if self.model_name in self.local_model_category:
            pass
        else:
            pass

    def teardown(self):
        if self.model_name in self.local_model_category:
            pass
        else:
            pass