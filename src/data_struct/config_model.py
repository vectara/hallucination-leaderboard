from pydantic import BaseModel
from typing import List


class ModelParams(BaseModel):
    """
    Parameters necssary for setting up a company specific model

    Fields
        model_name (str): name of model given by company
        date_code (str): date code of model, if doesnt exist input empty string
            ("")
    """
    model_name: str
    date_code: str
    class Keys:
        MODEL_NAME = "model_name"
        DATE_CODE = "date_code"

class ModelConfig(BaseModel):
    """
    Represents information necessary to initialize the correct object and 
    specifies if it should be used

    Fields
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
    #TODO: Doc
    pipeline: List[str]
    input_file: str
    temperature: float
    max_tokens: int
    simulation_count: int
    sample_count: int
    LLMs_to_eval: List[ModelConfig]

    class Keys:
        PIPELINE = "pipeline"
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