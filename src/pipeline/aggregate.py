import os
from datetime import datetime, timezone
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm

from .. analytics import (
    compute_hallucination_rate, compute_answer_rate,
    compute_avg_word_count, compute_confidence_interval
)
from .. LLMs import MODEL_REGISTRY
from .. data_model import Stats, EvalConfig
from .. json_utils import append_record_to_jsonl
from .. Logger import logger

"""
Aggregate the judgments on summaries to get per-LLM stats
"""

def aggregate_judgments(eval_config: EvalConfig):
    """
    Aggregate the judgments on summaries to get per-LLM stats

    Args:
        eval_config (EvalConfig): evaluation configuration
        judgment_file (str): name of the judgment file
        stats_file (str): name of the stats file

    Returns:
        None
    """

    judgment_file = eval_config.judgment_file
    stats_file = eval_config.stats_file

    LLMs_to_be_processed = [llm_config.model_name for llm_config in eval_config.per_LLM_configs]
    logger.info(f"Starting aggregation of summary-level judgments to per-LLM stats for the following LLMs: {LLMs_to_be_processed}")

    for llm_config in tqdm(eval_config.per_LLM_configs, desc="LLM Loop"):
        model_name = llm_config.model_name
        date_code = llm_config.date_code
        
        # Construct model output directory path
        model_out_dir = f"{eval_config.output_dir}/{llm_config.company}/{model_name}"

        logger.info(f"Generating results for {model_name}")

        JUDGMENT_CLASS = MODEL_REGISTRY[llm_config.company]["judgment_class"]

        judge_jsonl_path = os.path.join(model_out_dir, judgment_file)

        if os.path.isfile(judge_jsonl_path):
            logger.info(f"Judgment file {judgment_file} found for LLM {model_name}")

            results_jsonl_file = f"{stats_file}"
            results_jsonl_path = os.path.join(model_out_dir, results_jsonl_file)
            open(results_jsonl_path, 'w').close()
            generate_and_save_results(
                model_name, 
                judge_jsonl_path, 
                results_jsonl_path, 
                eval_config.eval_name, 
                eval_config.eval_date, 
                eval_config.hhem_version,
                JUDGMENT_CLASS,
                llm_config.date_code
            )
        else:
            logger.warning(
                f"Judgment file {judgment_file} not found for LLM {model_name}, skipping LLM"
            )
        logger.info(f"Finished aggregating summary-level judgments to per-LLM stats for LLM {model_name}")
    logger.info(f"Finished aggregating summary-level judgments to per-LLM stats for the following LLMs: {LLMs_to_be_processed}")

def generate_and_save_results(
        model_name: str, 
        judge_jsonl_path: str, 
        results_jsonl_path: str, 
        eval_name: str,
        eval_date: str,
        hhem_version: str,
        JUDGMENT_CLASS: type,
        date_code: str | None = None,
    ):
    """
    Loads per-summary judgments and aggregates them to get per-LLM stats.
    If date_code is provided, filters judgments by joining with summary data
    to get the date_code information.

    Args:
        model_name (str): name of model
        judge_jsonl_path (str): path to per-LLM judgments jsonl
        results_jsonl_path (str): path to new json file that saves the aggregated stats/metrics
        date_code (str | None): date code to filter by, if provided

    Returns:
        None
    """
    judgments_df = pd.read_json(judge_jsonl_path, lines=True)

    # If date_code is provided, we need to join with summary data to filter by date_code
    if date_code is not None:
        # Load summary data to get date_code information
        summary_file = "summaries.jsonl" #TOOD: load this from main.py
        summary_jsonl_path = os.path.join(os.path.dirname(judge_jsonl_path), summary_file)
        
        if os.path.isfile(summary_jsonl_path):
            summaries_df = pd.read_json(summary_jsonl_path, lines=True)
            summaries_df['date_code'] = summaries_df['date_code'].astype('string') # FIXME: use a schema-based approach to be more robust and apply to all columns
            # print(f"summaries_df: {summaries_df}")
            # Join judgments with summaries on summary_uid to get date_code
            judgments_df = pd.merge(
                judgments_df, 
                summaries_df[['summary_uid', 'date_code']], 
                on='summary_uid', 
                how='inner'
            )
            # print(f"judgments_df: {judgments_df}")
            print(f"judgments_df after merging: {judgments_df}", type(judgments_df))
            # print types of columns in judgments_df
            print(f"types of columns in judgments_df: {judgments_df.dtypes}")

            # Filter by date_code
            judgments_df = judgments_df[judgments_df['date_code'] == date_code]
            
            print("date_code: ", date_code, type(date_code))
        else:
            logger.warning(f"Summary file {summary_file} not found, cannot filter by date_code")

    # Add checking that the hhem_version passed in is the same as the hhem_version in the judgments_df
    if len(judgments_df) > 0 and hhem_version != judgments_df[JUDGMENT_CLASS.Keys.HHEM_VERSION].iloc[0]:
        logger.warning(f"HHEM version mismatch between passed-in hhem_version and hhem_version in judgments_df loaded from {judge_jsonl_path}")
    elif len(judgments_df) == 0:
        logger.warning(f"No judgments found after filtering by date_code {date_code} in {judge_jsonl_path}")

    hr = round(compute_hallucination_rate(judgments_df)*100.0, 1)
    ar = round(compute_answer_rate(judgments_df)*100.0, 1)
    awc = round(compute_avg_word_count(judgments_df), 1)
    ci = round(compute_confidence_interval(judgments_df)*100.0, 1)

    result_record = Stats(
        eval_name=eval_name,
        eval_date=eval_date,
        model_name=model_name,
        date_code=date_code,
        hhem_version=hhem_version,
        hallucination_rate=hr,
        confidence_interval=ci,
        answer_rate=ar,
        avg_word_count=awc
    )

    append_record_to_jsonl(results_jsonl_path, result_record)

    # Block below commented out because now date_code is part of the LLMConfig. But please keep the code below for future reference. -- Forrest, 2025-07-06

    # grouped_metric_df = metrics_df.groupby(Stats.Keys.DATE_CODE)

    # # One LLM can have multiple date codes, so we need to loop through each date code
    # for date_code, subset_df in tqdm(grouped_metric_df, total=len(grouped_metric_df), desc="Date Code Loop"):

    #     hr = round(compute_hallucination_rate(subset_df)*100.0, 1)
    #     ar = round(compute_answer_rate(subset_df)*100.0, 1)
    #     asw = round(compute_avg_summary_words(subset_df), 1)
    #     ci = round(compute_confidence_interval(subset_df)*100.0, 1)

    #     result_record = Stats(
    #         eval_name=eval_name,
    #         eval_date=eval_date,
    #         model_name=model_name,
    #         date_code=date_code,
    #         hhem_version=hhem_version,
    #         hallucination_rate=hr,
    #         confidence_interval=ci,
    #         answer_rate=ar,
    #         avg_summary_words=asw
    #     )

    #     append_record_to_jsonl(results_jsonl_path, result_record)