from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY, register_model
from src.data_struct.config_model import ExecutionMode, InteractionMode
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.exceptions import (
    ClientOrLocalNotInitializedError,
    ClientModelProtocolBranchNotFound,
    LocalModelProtocolBranchNotFound
)

COMPANY = "rednote"
@register_model(COMPANY)
class Rednote(AbstractLLM):
    """
    Class for models from rednote

    Attributes:
        client (Client): client associated with api calls with anthropic
        model (str): rednote style model name
    """

    local_models = ["rednote-hilab/dots.llm1.inst", "rednote-hilab/dots.llm1.base"]
    client_models = []

    model_category1 = ["rednote-hilab/dots.llm1.inst"]
    model_category2 = ["rednote-hilab/dots.llm1.base"]

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
        summary = EMPTY_SUMMARY
        if self.client_is_defined():
            if False:
                pass
            else:
                raise ClientModelProtocolBranchNotFound(self.model_name)
        elif self.local_model_is_defined():
            if self.model_name in self.model_category1:
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
                    outputs[0][input_tensor.shape[1]:],
                    skip_special_tokens=True
                )

                summary = result
            elif self.model_name in self.model_category2:
                tokenizer = AutoTokenizer.from_pretrained(self.model)


                inputs = tokenizer(prepared_text, return_tensors="pt")
                outputs = self.local_model.generate(
                    **inputs.to(self.local_model.device),
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True
                )
                result = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                summary = result
            else:
                raise LocalModelProtocolBranchNotFound(self.model_name)
        else:
            raise ClientOrLocalNotInitializedError(self.model_name)
        return summary

    def setup(self):
        if self.valid_client_model():
            pass
        elif self.valid_local_model():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.model,
            #     torch_dtype=torch.bfloat16 ,
                quantization_config=bnb_config
            ).to(self.device)

    def teardown(self):
        if self.client_is_defined():
            self.close_client()
        elif self.local_model_is_defined():
            self.default_local_model_teardown()

    def close_client(self):
        pass
