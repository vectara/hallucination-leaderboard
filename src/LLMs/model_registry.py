from src.LLMs.AbstractLLM import AbstractLLM

"""
Auto register that creates a dictionary mapping for all company model classes.

Global Variables:
    MODEL_REGISTRY (Dict): Dictionary containing company_name class_type pairs

Usage:
    Add @register_handle("company_name") to any new company model class
"""


MODEL_REGISTRY = {}

def register_model(company_name: str):
    """
    Decorater to auto-register a class under a give name

    Args:
        company_name: they key the class is registered under

    Returns:
        AbstractLLM: class associated with company_name
    """
    def decorator(company_class: AbstractLLM):
        MODEL_REGISTRY[company_name] = company_class
        return company_class
    return decorator