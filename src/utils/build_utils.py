from src.logging.Logger import logger
from src.LLMs.model_registry import MODEL_REGISTRY
from src.LLMs.AbstractLLM import AbstractLLM
from src.data_struct.config_model import ModelConfig
from pydantic import BaseModel, ValidationError

def builds_models(config: list[ModelConfig]) -> list[AbstractLLM]:
    #TODO: Update Doc
    """
    Given a config records, creates a list of model objects

    Args:
        config (list[dict]): list of dictionaries for model object init

    Returns:
        list[AbstractLLM]: list of models
    """

    models = []
    for model in config:

        if model.enabled:
            logger.log(f"{model.company}-{model.params.model_name}-{model.params.date_code} enabled")
        else:
            logger.log(f"{model.company}-{model.params.model_name}-{model.params.date_code} is disabled, skipping ")
            continue



        company_class = MODEL_REGISTRY.get(model.company)
        if company_class == None:
            logger.log("No registered class for this company, skipping")
            continue

        try:
            models.append(company_class(**model.params.dict()))
        except Exception as e:
            logger.log(f"Failed to instantiate {model.company}-{model.params.model_name}-{model.params.date_code} : {e}")

    return models

def process_raw_config(raw_model_configs):
    #TODO: Doc
    model_configs = []
    for i, raw_model_config in enumerate(raw_model_configs):
        try:
            model_config = ModelConfig(**raw_model_config)
            model_configs.append(model_config)
        except ValidationError as e:
            logger.log(f"Config error at index {i} with error {e}, replacing with dummy entry")
    return model_configs



# Keep for now but probably dont need, delete is fine
def find_valid_configs(model_configs: list[ModelConfig]):
    #TODO: Doc
    valid_configs = []
    for i in range(len(model_configs)):
        model_config = model_configs[i]
        if not all_fields_non_none(model_config):
            logger.log(f"Incomplete config for entry {i}, ignoring this entry")
        else:
            valid_configs.append(model_config)
    return valid_configs

# Keep for now but probably dont need, delete is fine
def all_fields_non_none(obj) -> bool:
    #TODO: Doc
    if isinstance(obj, BaseModel):
        return all(all_fields_non_none(value) for value in obj.__dict__.values())
    elif isinstance(obj, dict):
        return all(all_fields_non_none(v) for v in obj.values())
    elif isinstance(obj, list):
        return all(all_fields_non_none(i) for i in obj)
    else:
        return obj is not None