from src.LLMs.AbstractLLM import AbstractLLM
from huggingface_hub import InferenceClient
import re
import time

class DeepSeekAI(AbstractLLM):

    def __init__(self, model_name, data_code=None):
        super().__init__(model_name=model_name, company="deepseek-ai")
        self.full_model_name = f"{self.company}/{self.model_name}"
        self.client = InferenceClient(model=self.full_model_name)

    def summarize(self, prepared_text: str) -> str:
        messages = [{"role": "user", "content":prepared_text}]
        client_package = self.client.chat_completion(messages, temperature=self.temperature)
        raw_summary = client_package.choices[0].message.content
        summary = re.sub(r'<think>.*?</think>\s*', '', raw_summary, flags=re.DOTALL)
        time.sleep(4) # Avoid Throttling
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass