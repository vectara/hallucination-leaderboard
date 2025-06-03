from src.logging.Logger import logger
from src.LLMs.model_registry import MODEL_REGISTRY
from src.LLMs.AbstractLLM import AbstractLLM

def builds_models(config: list[dict]) -> list[AbstractLLM]:
    """
    Given a config records, creates a list of model objects

    Args:
        config (list[dict]): list of dictionaries for model object init

    Retunrs:
        list[AbstractLLM]: list of models
    """

    models = []
    for entry in config:
        company = entry.get("company")
        params = entry.get("params", {})

        if not company:
            logger.log("Missing Company key, skipping")
            continue

        model_class = MODEL_REGISTRY.get(company)
        if not model_class:
            logger.log("No registered model for this company, skipping")
            continue
            
        print(f"adding {company}")

        try:
            models.append(model_class(**params))
        except Exception as e:
            logger.log(f"Failed to instantiate {company} model: {e}")

    return models