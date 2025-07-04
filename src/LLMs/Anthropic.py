# from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY, MODEL_REGISTRY


import os
import anthropic

from .AbstractLLM import AbstractLLM, BasicLLMConfig
from .. data_model import Summary, ModelInstantiationError

# Forrest, 2025-07-02
# from src.exceptions import (
#         ClientOrLocalNotInitializedError,
#         ClientModelProtocolBranchNotFound,
#         LocalModelProtocolBranchNotFound
#     )

COMPANY = "anthropic"

class AnthropicConfig(BasicLLMConfig):
    pass 

class AnthropicSummary(Summary):
    pass

class Anthropic(AbstractLLM):
    """
    Class for models from Anthropic

    Attributes:
        client (Client): client associated with api calls with anthropic
        model (str): anthropic style model name
    """

    # local_models = []
    # client_models = ["claude-opus-4", "claude-sonnet-4"]
    # model_category1 = ["claude-opus-4", "claude-sonnet-4"]

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "claude-opus-4": 1,
        "claude-sonnet-4": 1
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {   
    }

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

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name]:
                case 1:
                    chat_package = self.client.messages.create(
                        model=self.model,
                        messages=[{"role": "user", "content":prepared_text}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )   
                    summary = chat_package.content[0].text
        elif self.local_model: 
            pass
        else:
            raise ModelInstantiationError.MISSING_SETUP(self.__class__.__name__)
        # elif self.local_model_is_defined():
        #     if False:
        #         pass
        #     else:
        #         raise LocalModelProtocolBranchNotFound(self.model_name)
        # else:
        #     raise ClientOrLocalNotInitializedError(self.model_name)
        return summary

    # def setup(self):
    #     if self.valid_client_model():
    #         api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
    #         self.client = anthropic.Client(api_key=api_key)
    #     elif self.valid_local_model():
    #         pass

    def setup(self):
        if self.execution_mode == ExecutionMode.CLIENT:
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                self.client = anthropic.Client(api_key=api_key)
            else:
                raise ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                )
        elif self.execution_mode == ExecutionMode.LOCAL:
            pass

    def teardown(self):
        if self.client_is_defined():
            self.close_client()
        elif self.local_model_is_defined():
            self.default_local_model_teardown()

    def close_client(self):
        pass

# MODEL_REGISTRY[COMPANY] = Anthropic
