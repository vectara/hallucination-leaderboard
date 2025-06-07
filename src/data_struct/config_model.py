from pydantic import BaseModel

#TODO: Doc

class ModelParams(BaseModel):
    model_name: str
    date_code: str
    class Keys:
        MODEL_NAME = "model_name"
        DATE_CODE = "date_code"

class ModelConfig(BaseModel):
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