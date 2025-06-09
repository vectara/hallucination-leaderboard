from pydantic import BaseModel

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
    enabled: bool
    params: ModelParams

    class Keys:
        COMPANY = "company"
        ENABLED = "enabled"
        PARAMS = "params"

DUMMY_CONFIG = ModelConfig(
    company="INVALID",
    enabled=False,
    params=ModelParams(
        model_name="INVALID",
        date_code="INVALID"
    )
)