import pandas as pd
from src.data_struct.data_model import Judgement

def compute_hallucination_rate(
        metrics_df: pd.DataFrame, threshold=0.5
    ) -> float:
    """
    Computes hallucination rate with default threshold of 0.5

    Args:
        metrics_df (pd.DataFrame): metrics dataframe
        threshold (float): confidence threshold for positive result

    Returns:
        float: hallucination rate
    """

    fcr = compute_factual_consistancy_rate(
        metrics_df, threshold=threshold
    )
    hallucination_rate = 1.0 - fcr
    return hallucination_rate

def compute_factual_consistancy_rate(
        metrics_df: pd.DataFrame, threshold=0.5
    ) -> float:
    """
    Computes factual consistancy rate with default threshold of 0.5

    Args:
        metrics_df (pd.DataFrame): metrics dataframe
        threshold (float): confidence threshold for positive result

    Returns:
        float: factual consistancy rate
    """

    valid_summs_df = metrics_df[metrics_df[Judgement.Keys.VALID]]
    if valid_summs_df.empty:
        return 0.0
    total_count = valid_summs_df.shape[0]

    factual_count = 0
    for score in valid_summs_df[Judgement.Keys.HHEM_SCORE].tolist():
        if score >= threshold:
            factual_count += 1
    factual_consistancy_rate = factual_count/total_count
    return factual_consistancy_rate

def compute_confidence_interval(metrics_df: pd.DataFrame) -> float:
    return -0.01

def compute_answer_rate(metrics_df: pd.DataFrame) -> float:
    """
    Computes the the rate of valid summaries. is_valid_summary in metrics.py
    for definition of a valid summary.

    Args:
        metrics_df (pd.DataFrame): metrics dataframe

    Returns:
        float: answer rate
    """
    if metrics_df[Judgement.Keys.VALID].empty:
        return 0.0
    answer_rate = metrics_df[Judgement.Keys.VALID].mean()
    return answer_rate

def compute_avg_summary_length(metrics_df: pd.DataFrame) -> float:
    """
    Computes average summary length

    Args:
        metrics_df (pd.DataFrame): metrics dataframe

    Returns:
        float: Average summary length
    """

    valid_summs_df = metrics_df[metrics_df[Judgement.Keys.VALID]]
    if valid_summs_df.empty:
        return 0.0
    avg_summary_length = valid_summs_df[Judgement.Keys.SUMMARY_WORDS].mean()
    return avg_summary_length