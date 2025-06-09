import pandas as pd
from src.LLMs.AbstractLLM import (
    MODEL_RETURNED_NON_STRING_TYPE_OUTPUT, MODEL_FAILED_TO_RETURN_OUTPUT
)


def is_valid_summary(summary: str):
    """
    Checks if summary is valid and returns True if it is else False

    Args:
        summary (str): the summary

    Returns:
        bool: True if valid summary else False
    """

    if has_error_output(summary):
        return False
    elif len(summary.split()) >= 5:
        return True
    else:
        return False

def has_error_output(summary: str):
    """
    Detects if summary contains error output and returns True if so

    Args:
        summary (str): the summary

    Returns:
        bool: True if summary is exact error output string
    """

    if (
        summary == MODEL_FAILED_TO_RETURN_OUTPUT or
        summary == MODEL_RETURNED_NON_STRING_TYPE_OUTPUT
    ):
        return True
    else:
        return False