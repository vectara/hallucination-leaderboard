from src.Logger import logger
from src.LLMs.model_registry import MODEL_REGISTRY
from src.LLMs.AbstractLLM import AbstractLLM
from src.data_struct.config_model import ModelConfig, Config
from pydantic import ValidationError

"""
Functions for building models from the config.json

Functions:
    buld_models(config)
    process_raw_config(raw_model_configs)
"""

def builds_models(config: list[ModelConfig]) -> list[AbstractLLM]:
    """
    Builds the models given in the config list if it is registered

    Args:
        config (list[dict]): list of dictionaries for model object init

    Returns:
        list[AbstractLLM]: list of models
    """

    models = []
    for model in config:
        company_class = MODEL_REGISTRY.get(model.company)
        if company_class == None:
            logger.warning("No registered class for this company, skipping")
            print(f"This {company_class} is not registered, can't build")
            continue

        try:
            models.append(company_class(**model.params.model_dump()))
        except Exception as e:
            logger.warning(
                f"failed to instantiate {model.company}-"
                f"{model.params.model_name}-{model.params.date_code} : {e}"
            )
            print(
                f"failed to instantiate {model.company}-"
                f"{model.params.model_name}-{model.params.date_code} : {e}"
            )
    return models

def process_raw_config(raw_model_configs: list[dict]) -> list[ModelConfig]:
    """
    Converts raw config json data into a list of ModelConfig objects

    Args:
        raw_model_configs (list[dict]):

    Returns:
        list[ModelConfig]
    
    """
    model_configs = []
    for i, raw_model_config in enumerate(raw_model_configs):
        try:
            model_config = ModelConfig(**raw_model_config)
            model_configs.append(model_config)
        except ValidationError as e:
            logger.warning(
                f"Config error at index {i} with error {e}, replacing with "
                "dummy entry"
            )
    return model_configs
