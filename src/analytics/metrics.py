from src.LLMs.AbstractLLM import SUMMARY_ERRORS


def is_valid_summary(summary: str) -> bool:
    """
    Checks if summary is valid and returns True if it is else False. A summary
    is valid if it is longer than 4 words and the output is not equivalent to
    specific error strings

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

def has_error_output(summary: str) -> bool:
    """
    Detects if summary contains error output and returns True if so

    Args:
        summary (str): the summary

    Returns:
        bool: True if summary is exact error output string
    """

    if summary in SUMMARY_ERRORS:
        return True
    else:
        return False