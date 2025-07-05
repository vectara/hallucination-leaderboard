import pandas as pd

from . data_model import Judgement, SummaryError

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

    # Get all SummaryError values
    error_values = [error.value for error in SummaryError]
    
    if summary in error_values:
        return False
    elif len(summary.split()) < 5:
        return False
    else:
        return True

# def has_error_output(summary: str) -> bool:
#     """
#     Detects if summary contains error output and returns True if so

#     Args:
#         summary (str): the summary

#     Returns:
#         bool: True if summary is exact error output string
#     """

#     if summary in SUMMARY_ERRORS:
#         return True
#     else:
#         return False


def compute_hallucination_rate(
        metrics_df: pd.DataFrame, threshold=0.5
    ) -> float:
    """
    Computes hallucination rate for valid summaries with default threshold of 
    0.5. If factual conistancy rate returns -1.0 this also returns -1.0. 
    -1.0 means there were no valid summaries in metrics_df.

    Args:
        metrics_df (pd.DataFrame): metrics dataframe
        threshold (float): confidence threshold for positive result

    Returns:
        float: hallucination rate
    """

    fcr = compute_factual_consistancy_rate(
        metrics_df, threshold=threshold
    )
    if fcr == -1.0:
        return -1.0

    hallucination_rate = 1.0 - fcr
    return hallucination_rate

def compute_factual_consistancy_rate(
        metrics_df: pd.DataFrame, threshold=0.5
    ) -> float:
    """
    Computes factual consistancy rate for valid summaries with default threshold
    of 0.5. Assigned a value of -1.0 when metrics_df has no valid summaries.

    Args:
        metrics_df (pd.DataFrame): metrics dataframe
        threshold (float): confidence threshold for positive result

    Returns:
        float: factual consistancy rate
    """

    valid_summs_df = metrics_df[metrics_df[Judgement.Keys.VALID]]
    if valid_summs_df.empty:
        return -1.0
    total_count = valid_summs_df.shape[0]

    factual_count = 0
    for score in valid_summs_df[Judgement.Keys.HHEM_SCORE].tolist():
        if score >= threshold:
            factual_count += 1
    factual_consistancy_rate = factual_count/total_count
    return factual_consistancy_rate

def compute_confidence_interval(metrics_df: pd.DataFrame) -> float:
    """
    Not defined yet, gives a dummy value for now.
    """
    return -0.01

def compute_answer_rate(metrics_df: pd.DataFrame) -> float:
    """
    Computes the the rate of valid summaries. Returns -1.0 if there are no valid
    summaries in metrics_df.

    Args:
        metrics_df (pd.DataFrame): metrics dataframe

    Returns:
        float: answer rate
    """
    if metrics_df[Judgement.Keys.VALID].empty:
        return -1.0
    answer_rate = metrics_df[Judgement.Keys.VALID].mean()
    return answer_rate

def compute_avg_summary_length(metrics_df: pd.DataFrame) -> float:
    """
    Computes average summary length for valid summaries only. Returns -1.0 if
    there are no valid summaries.

    Args:
        metrics_df (pd.DataFrame): metrics dataframe

    Returns:
        float: Average summary length
    """

    valid_summs_df = metrics_df[metrics_df[Judgement.Keys.VALID]]
    if valid_summs_df.empty:
        return -1.0
    avg_summary_length = valid_summs_df[Judgement.Keys.SUMMARY_WORDS].mean()
    return avg_summary_length 