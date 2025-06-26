from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
import os
import anthropic
from src.LLMs.model_registry import register_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

COMPANY = "rednote"
@register_model(COMPANY)
class Rednote(AbstractLLM):
    """
    Class for models from rednote

    Attributes:
        client (Client): client associated with api calls with anthropic
        model (str): rednote style model name
    """

    local_model = ["rednote-hilab/dots.llm1.inst"]
    client_model = []

    model_category1 = ["rednote-hilab/dots.llm1.inst"]

    def __init__(
            self,
            model_name: str,
            execution_mode: str,
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
        if self.local_model and self.model in self.model_category1:
            tokenizer = AutoTokenizer.from_pretrained(self.model)

            input_tensor = tokenizer.apply_chat_template(
                {"role": "user", "content": prepared_text},
                add_generation_prompt=True,
                return_tensors="pt"
            )

            outputs = self.local_model.generate(
                input_tensor.to(self.local_model.device),
                max_new_tokens=self.max_tokens
            )

            result = tokenizer.decode(
                outputs[0][input_tensor.shape[1]:], skip_special_tokens=True
            )

            summary = result
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return summary

    def setup(self):
        if self.valid_client_model():
            pass
        elif self.valid_local_model():
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.model, device_map="auto", torch_dtype=torch.bfloat16
            )
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
