
import os
from typing import Literal


from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

COMPANY = "alibaba"
class AlibabaConfig(BasicLLMConfig):
    """Extended config for Alibaba-specific properties"""
    company: Literal["alibaba"] 
    model_name: Literal["Qwen3-32B"] # Only model names manually added to this list are supported.
    date_code: str # do we need date code for ali baba?
    execution_mode: Literal["api"] = "api" # Is Alibaba only API based?

class AlibabaSummary(BasicSummary):
    pass # Nothing additional to the BasicSummary class.

class AlibabaLLM(AbstractLLM):
    """
    Class for models from Alibaba
    """

    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {}

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: AlibabaConfig):
        super().__init__(config)

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match self.client_mode_group[self.model_name]:
                case 1:
                    summary = None
        elif self.local_model: 
            pass
        else:
            raise Exception(
                ModelInstantiationError.MISSING_SETUP.format(
                    class_name=self.__class__.__name__
                )
            )
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, f"Alibaba API key not found in environment variable {COMPANY.upper()}_API_KEY"
                self.client = None
            else:
                raise Exception(ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                    model_name=self.model_name,
                    company=self.company,
                    execution_mode=self.execution_mode
                ))
        elif self.execution_mode == "local":
            pass

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass