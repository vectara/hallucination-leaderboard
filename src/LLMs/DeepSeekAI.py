from src.LLMs.AbstractLLM import AbstractLLM
from huggingface_hub import InferenceClient
import re
import time

class DeepSeekAI(AbstractLLM):

    def __init__(self, model_name, data_code=None):
        super().__init__(model_name=model_name, company="deepseek-ai")
        self.full_model_name = f"{self.company}/{self.model_name}"
        self.client = InferenceClient(model=self.full_model_name)
        self.min_throttle_time = 4 # 1000 req per hour with hf pro, avoid throttle time

    def summarize(self, prepared_text: str) -> str:
        messages = [{"role": "user", "content":prepared_text}]
        start_time = time.time()
        client_package = self.client.chat_completion(messages, temperature=self.temperature)
        elapsed_time = time.time() - start_time
        remaining_time = self.min_throttle_time - elapsed_time
        if remaining_time > 0: # Delay only if it took longer than 4s
            time.sleep(remaining_time)
        raw_summary = client_package.choices[0].message.content
        summary = re.sub(r'<think>.*?</think>\s*', '', raw_summary, flags=re.DOTALL)
        return summary

    def setup(self):
        pass

    def teardown(self):
        pass