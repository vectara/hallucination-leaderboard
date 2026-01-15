"""Analytics functions for computing evaluation metrics and statistics.

This module provides functions for validating summaries and computing
aggregate metrics from judgment data. Includes hallucination rate,
factual consistency rate, answer rate, and word count statistics.

Functions:
    is_valid_summary: Check if a summary meets validity criteria.
    compute_hallucination_rate: Calculate percentage of hallucinated summaries.
    compute_factual_consistancy_rate: Calculate percentage of factually consistent summaries.
    compute_confidence_interval: Calculate confidence interval for metrics.
    compute_answer_rate: Calculate percentage of valid summaries.
    compute_avg_word_count: Calculate average word count for valid summaries.
    clean_string: Preprocess text for improved HHEM performance.
"""

import pandas as pd
import re
import string

from . data_model import BasicJudgment, SummaryError

def is_valid_summary(summary: str) -> bool:
    """Check if a summary meets validity criteria.

    A summary is considered valid if it contains more than 5 words and
    does not match any predefined error strings from the SummaryError enum.

    Args:
        summary: The summary text to validate.

    Returns:
        True if the summary is valid, False otherwise.

    Note:
        This validation may not catch all invalid summaries, such as
        model refusals or explanations of why content cannot be summarized.
    """
    # Get all SummaryError values
    error_values = [error.value for error in SummaryError]

    # FIXME: This needs to be expanded. What if the LLM says I am sorry that I cannot answer that query? 
    # For example, Anthropic may say: "I cannot provide a summary of the passage because no actual content was provided. The text only contains the phrase \"Dummy article with no content,\" which indicates there is no substantive information to summarize. To provide a meaningful summary, I would need a passage that contains actual content and information."
    # As another example, OpenAI may say: ""The passage does not contain any information to summarize."


    
    if summary.strip() in error_values:
        return False
    elif len(summary.split()) <= 5:
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
    """Compute the hallucination rate for valid summaries.

    Calculates the proportion of valid summaries that are classified as
    hallucinated based on the HHEM score threshold. Hallucination rate
    is the complement of factual consistency rate.

    Args:
        metrics_df: DataFrame containing judgment data with HHEM scores
            and validity flags.
        threshold: HHEM score threshold for classifying a summary as
            factually consistent. Defaults to 0.5.

    Returns:
        Hallucination rate as a float between 0.0 and 1.0, or -1.0 if
        there are no valid summaries in the DataFrame.
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
    """Compute the factual consistency rate for valid summaries.

    Calculates the proportion of valid summaries with HHEM scores at or
    above the specified threshold, indicating factual consistency with
    the source document.

    Args:
        metrics_df: DataFrame containing judgment data with HHEM scores
            and validity flags.
        threshold: HHEM score threshold for classifying a summary as
            factually consistent. Defaults to 0.5.

    Returns:
        Factual consistency rate as a float between 0.0 and 1.0, or -1.0
        if there are no valid summaries in the DataFrame.
    """
    valid_summs_df = metrics_df[metrics_df[BasicJudgment.Keys.IS_VALID]]
    if valid_summs_df.empty:
        return -1.0
    total_count = valid_summs_df.shape[0]

    factual_count = 0
    for score in valid_summs_df[BasicJudgment.Keys.HHEM_SCORE].tolist():
        if score >= threshold:
            factual_count += 1
    factual_consistancy_rate = factual_count/total_count
    return factual_consistancy_rate

def compute_confidence_interval(metrics_df: pd.DataFrame) -> float:
    """Compute the confidence interval for hallucination rate.

    Placeholder function for computing statistical confidence intervals
    around the hallucination rate metric.

    Args:
        metrics_df: DataFrame containing judgment data.

    Returns:
        Confidence interval value, currently returns -0.01 as a placeholder.

    Todo:
        Implement proper confidence interval calculation using binomial
        proportion confidence interval or bootstrap methods.
    """
    return -0.01

def compute_answer_rate(metrics_df: pd.DataFrame) -> float:
    """Compute the answer rate (proportion of valid summaries).

    Calculates the proportion of summaries that pass validity checks,
    indicating the model successfully produced a substantive response
    rather than an error or refusal.

    Args:
        metrics_df: DataFrame containing judgment data with validity flags.

    Returns:
        Answer rate as a float between 0.0 and 1.0, or -1.0 if the
        validity column is empty.
    """
    if metrics_df[BasicJudgment.Keys.IS_VALID].empty:
        return -1.0
    answer_rate = metrics_df[BasicJudgment.Keys.IS_VALID].mean()
    return answer_rate

def compute_avg_word_count(metrics_df: pd.DataFrame) -> float:
    """Compute the average word count for valid summaries.

    Calculates the mean word count across all summaries that pass
    validity checks, providing insight into summary verbosity.

    Args:
        metrics_df: DataFrame containing judgment data with word counts
            and validity flags.

    Returns:
        Average word count as a float, or -1.0 if there are no valid
        summaries in the DataFrame.
    """
    valid_summs_df = metrics_df[metrics_df[BasicJudgment.Keys.IS_VALID]]
    if valid_summs_df.empty:
        return -1.0
    avg_word_count = valid_summs_df[BasicJudgment.Keys.WORD_COUNT].mean()
    return avg_word_count

def clean_string(s: str) -> str:
    """Clean and normalize text for improved HHEM performance.

    Applies preprocessing steps to standardize text input before
    hallucination detection. Currently performs basic whitespace
    trimming, with additional cleaning steps commented out for
    potential future use.

    Args:
        s: The input string to clean.

    Returns:
        The cleaned and normalized string.

    Note:
        Additional cleaning options (citation removal, bracket handling,
        punctuation normalization) are available but currently disabled.
    """
    s = s.strip()
    # remove citations in the form of [1], [2], etc.
    # s = re.sub(r'\[\d+\]', '', s)
    # for any square brackets, remove the brackets but keep the content
    # s = re.sub(r'\[(.*?)\]', r'\1', s)
    # remove special characters
    # s = re.sub(r'[^\w\s]', '', s)
    # remove extra whitespace
    # s = re.sub(r'\s+', ' ', s)
    # remove spaces before any punctuation
    # s = re.sub(r'\s([.,!?])', r'\1', s)
    # if the string does not end with a punctuation, add a period
    # if not any([s.endswith(p) for p in string.punctuation]):
    #    s = s + '.'
    return s