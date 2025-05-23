from src.LLMs.AbstractLLM import AbstractLLM
import anthropic
import os

class ClaudeOpus4p0(AbstractLLM):

    def __init__(self):
        super().__init__("Anthropic_Claude-Opus-4.0")
        api_key = os.getenv("ANTHROPIC_KEY")
        self.client = anthropic.Client(api_key=api_key)
        self.model_type = "claude-4-opus-20250514"

    def summarize(self, prepared_text: str) -> str:
        chat_package = self.client.messages.create(
            model=self.model_type,
            messages=[{"role": "user", "content":prepared_text}],
            max_tokens=1024 #TODO: This should be constant for all models
        )
        summary = chat_package.content[0].text
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass