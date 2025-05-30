from src.LLMs.AbstractLLM import AbstractLLM
import os
import anthropic

class Anthropic(AbstractLLM):
    claude_4 = ["claude-4-opus", "claude-4-sonnet"]
    def __init__(self, model_name, data_code=None):
        super().__init__(model_name=model_name, company="anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if data_code:
            self.full_model_name = f"{self.company}/{self.model_name}-{data_code}"
        else:
            self.full_model_name = f"{self.company}/{self.model_name}"
        self.client = anthropic.Client(api_key=api_key)
        self.model = f"{model_name}-{data_code}"

    def summarize(self, prepared_text: str) -> str:
        summary = None
        if self.model_name in self.claude_4:
            chat_package = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content":prepared_text}],
                max_tokens=self.max_tokens
            )
            summary = chat_package.content[0].text
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass