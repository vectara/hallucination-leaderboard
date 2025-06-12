from src.LLMs.AbstractLLM import AbstractLLM
import os
import anthropic
from src.LLMs.model_registry import register_model

@register_model("anthropic")
class Anthropic(AbstractLLM):
    """
    Class for models from Anthropic

    Class Attributes:
        claude_4: list of claude_4 tier models that follow the same protocol
            for getting a summary

    Attributes:
        client (Client): client associated with api calls with anthropic
        model (str): anthropic style model name
    """

    claude_4 = ["claude-4-opus", "claude-4-sonnet"]
    def __init__(self, model_name, date_code=""):
        super().__init__(model_name=model_name, company="anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key=api_key)
        self.model = self.setup_model_identifier(model_name, date_code)

    def summarize(self, prepared_text: str) -> str:
        summary = None
        if self.model_name in self.claude_4:
            chat_package = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            summary = chat_package.content[0].text
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass