from pydantic import BaseModel
from typing import List, Optional, Literal
from enum import Enum

class ExecutionMode(str, Enum):
    CLIENT = "client"
    LOCAL = "local"

class ModelParams(BaseModel):
    """
    Parameters necessary for setting up a company specific model

    Fields:
        model_name (str): name of model given by company
        execution_mode (str): method of execution. Only accepts client or local
        date_code (str): date code of model. Optional, defaults to ""
        temperature (float): temperature hyperparm for model. Optional, 
            defaults to 0.0
        max_tokens (int): number of allowed output tokens
        thinking_tokens (int): number of thinking tokens for reasoning models
        min_throttle_time (float): time ine seconds required for each request
            to avoid throttling
    """

    model_name: str
    execution_mode: ExecutionMode
    date_code: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    thinking_tokens: int = 0
    min_throttle_time: float = 0.0

    class Keys:
        MODEL_NAME = "model_name"
        EXECUTION_MODE = "execution_mode"
        DATE_CODE = "date_code"
        TEMPERATURE = "temperature"
        MAX_TOKENS = "max_tokens"
        THINKING_TOKENS = "thinking_tokens"
        MIN_THROTTLE_TIME = "min_throttle_time"

class ModelConfig(BaseModel):
    """
    Represents information necessary to initialize the correct object and 
    specifies if it should be used

    Fields:
        company (str): company model belongs to, spelling has to be identical
            to how we label it in our model registry
        enabled (bool): if false model is ignored during run time of program
        params (ModelParams): parameters for model initialization
    """
    company: str
    params: ModelParams

    class Keys:
        COMPANY = "company"
        PARAMS = "params"

class Config(BaseModel):
    """
    Represents program configuration. Includes information from protocol to run 
    to all models that will run and their various hyperparameters.

    Fields:
        pipeline (list[str]): list of defined protocols, protocols are ran in 
            sequence of the list
        overwrite (Bool): if true overwrites the summary file before adding new
            data
        input_file (str): path to file that contains a dataset of articles that
            will be summarized by the LLMs
        temperature (float): hyperparameter that is given to all LLMs
        max_tokens (int): hyperparamter that is given to all LLMs
        simulation_count (int): number of times a summary will be generated for
            the entire dataset
        sample_count (int): number of samples from simulations
        LLMs_to_eval (list[ModelConfig]): list of model configuration 
            representations
    """
    pipeline: List[str]
    overwrite: bool
    input_file: str
    temperature: float # Default 0.0
    max_tokens: int # Default 1024
    simulation_count: int
    sample_count: int
    LLMs_to_eval: List[ModelConfig]

    def model_post_init(self, __context):
        for model_config in self.LLMs_to_eval:
            if model_config.params.temperature is None:
                model_config.params.temperature = self.temperature
            if model_config.params.max_tokens is None:
                model_config.params.max_tokens = self.max_tokens

    class Keys:
        PIPELINE = "pipeline"
        OVERWRITE = "overwrite"
        INPUT_FILE = "input_file"
        TEMPERATURE = "temperature"
        MAX_TOKENS = "max_tokens"
        SIMULATION_COUNT = "simulation_count"
        SAMPLE_COUNT = "sample_count"
        LLMS_TO_EVAL = "LLMs_to_eval"