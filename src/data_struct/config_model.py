from pydantic import BaseModel
from typing import List, Optional


class ModelParams(BaseModel):
    """
    Parameters necessary for setting up a company specific model

    Fields:
        model_name (str): name of model given by company
        temperature (float): temperature hyperparm for model. Optional, 
            defaults to 0.0
        date_code (str): date code of model. Optional, defaults to ""
    """
    model_name: str
    temperature: float = 0.0
    date_code: str = ""
    class Keys:
        MODEL_NAME = "model_name"
        TEMPERATURE = "temperature"
        DATE_CODE = "date_code"

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
    temperature: float
    max_tokens: int
    simulation_count: int
    sample_count: int
    LLMs_to_eval: List[ModelConfig]

    def model_post_init(self, __context):
        for model_config in self.LLMs_to_eval:
            model_config.params.temperature = self.temperature

    class Keys:
        PIPELINE = "pipeline"
        OVERWRITE = "overwrite"
        INPUT_FILE = "input_file"
        TEMPERATURE = "temperature"
        MAX_TOKENS = "max_tokens"
        SIMULATION_COUNT = "simulation_count"
        SAMPLE_COUNT = "sample_count"
        LLMS_TO_EVAL = "LLMs_to_eval"

DUMMY_CONFIG = ModelConfig(
    company="INVALID",
    enabled=False,
    params=ModelParams(
        model_name="INVALID",
        date_code="INVALID"
    )
)