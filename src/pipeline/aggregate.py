"""Aggregation pipeline for computing per-LLM statistics from judgments.

This module provides functionality for aggregating summary-level judgment
scores into per-LLM statistics. Computes metrics such as hallucination rate,
answer rate, average word count, and confidence intervals for each model.

The pipeline groups judgments by summary date and judgment date, enabling
tracking of model performance over time and across evaluation runs.

Functions:
    aggregate_judgments: Main entry point for aggregating judgment results.
    generate_and_save_results: Core aggregation logic for a single LLM.
"""

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
from .. data_model import Stats, EvalConfig, BasicLLMConfig, BasicJudgment
from .. json_utils import append_record_to_jsonl
from .. Logger import logger

def aggregate_judgments(eval_config: EvalConfig):
    """Aggregate summary-level judgments into per-LLM statistics.

    Main entry point for the aggregation pipeline. Iterates through all
    configured LLMs, loads their judgment files, and computes aggregate
    statistics including hallucination rate, answer rate, and confidence
    intervals. Results are saved to stats JSONL files.

    Args:
        eval_config: Evaluation configuration containing LLM configs, file
            paths, and evaluation metadata.

    Note:
        Skips LLMs whose judgment files are not found, logging a warning
        for each missing file. Common LLM config settings are merged into
        per-LLM configs before processing.
    """
    judgment_file = eval_config.judgment_file
    stats_file = eval_config.stats_file

    LLMs_to_be_processed = [
        llm_config.model_name for llm_config in eval_config.per_LLM_configs
    ]
    logger.info(
        f"Starting aggregation of summary-level judgments to per-LLM stats for "
        f"the following LLMs: {LLMs_to_be_processed}"
        )

    for llm_config in tqdm(eval_config.per_LLM_configs, desc="LLM Loop"):
        llm_alias = ""
        if llm_config.date_code == "" or llm_config.date_code == None:
            llm_alias = f"{llm_config.model_name}"
        else:
            llm_alias = f"{llm_config.model_name}-{llm_config.date_code}"

        for common_key in eval_config.common_LLM_config.model_fields_set:
            if common_key not in llm_config.model_fields_set:
                llm_config = llm_config.model_copy(
                    update={
                        common_key: getattr(
                            eval_config.common_LLM_config,
                            common_key
                        )
                    }
                )
        model_name = llm_config.model_name
        
        model_out_dir = ""
        if llm_config.date_code == "" or llm_config.date_code == None:
            model_out_dir = os.path.join(
                eval_config.output_dir, 
                llm_config.company, 
                llm_config.model_name
            )
        else:
            model_out_dir = os.path.join(
                eval_config.output_dir, 
                llm_config.company, 
                f"{llm_config.model_name}-{llm_config.date_code}"
            )

        logger.info(f"Aggregating judgements to stats for {llm_alias}")

        judgments_jsonl_path = os.path.join(model_out_dir, judgment_file)

        if os.path.isfile(judgments_jsonl_path):
            logger.info(
                f"Judgment file {judgment_file} found for LLM {model_name}"
            )

            results_jsonl_path = os.path.join(model_out_dir, stats_file)
            open(results_jsonl_path, 'w').close()

            generate_and_save_results(eval_config, llm_config)
        else:
            logger.warning(
                f"Judgment file {judgment_file} not found for LLM {llm_alias}, "
                f"skipping LLM"
            )
        logger.info(
            f"Finished aggregating summary-level judgments to per-LLM stats "
            f"for LLM {llm_alias}"
        )
    logger.info(
        f"Finished aggregating summary-level judgments to per-LLM stats for "
        f"the following LLMs: {LLMs_to_be_processed}"
    )

def generate_and_save_results(
        eval_config: EvalConfig,
        llm_config: BasicLLMConfig,
    ):
    """Load judgments and compute aggregated statistics for a single LLM.

    Loads per-summary judgment scores, optionally filters by date_code,
    groups by summary and judgment dates, and computes aggregate metrics
    for each group. Results are saved incrementally to a stats JSONL file.

    Computed metrics include:
        - Hallucination rate (percentage)
        - Answer rate (percentage)
        - Average word count
        - Confidence interval (percentage)

    Args:
        eval_config: Evaluation configuration with file paths and settings.
        llm_config: Model-specific configuration including company, model
            name, and optional date_code for filtering.

    Note:
        If date_code is provided, judgments are filtered by joining with
        summary data to match the specified date_code. Logs warnings for
        HHEM version mismatches and empty result sets after filtering.
    """
    date_code = llm_config.date_code
    hhem_version = eval_config.hhem_version

    LLM_SUMMARY_CLASS = MODEL_REGISTRY[llm_config.company]["summary_class"]
    SUMMARY_MODEL_AS_DICT: Dict[str, type] = {
        field_name: field_type.annotation for field_name,
        field_type in LLM_SUMMARY_CLASS.model_fields.items()
    }
    JUDGMENT_MODEL_AS_DICT: Dict[str, type] = {
        field_name: field_type.annotation for field_name,
        field_type in BasicJudgment.model_fields.items()
    }

    model_out_dir = ""
    if llm_config.date_code == "" or llm_config.date_code == None:
        model_out_dir = os.path.join(
            eval_config.output_dir, 
            llm_config.company, 
            llm_config.model_name
        )
    else:
        model_out_dir = os.path.join(
            eval_config.output_dir, 
            llm_config.company, 
            f"{llm_config.model_name}-{llm_config.date_code}"
        )
    summaries_jsonl_path = os.path.join(
        model_out_dir,
        eval_config.summary_file
    )
    judgments_jsonl_path = os.path.join(
        model_out_dir,
        eval_config.judgment_file)
    stats_jsonl_path = os.path.join(model_out_dir, eval_config.stats_file)
    judgments_df = pd.read_json(
        judgments_jsonl_path,
        lines=True,
        dtype=JUDGMENT_MODEL_AS_DICT
    )
    
    if os.path.isfile(summaries_jsonl_path):
        summaries_df = pd.read_json(
            summaries_jsonl_path,
            lines=True,
            dtype=SUMMARY_MODEL_AS_DICT
        )

        judgments_df = pd.merge(
            judgments_df, 
            summaries_df[['summary_uid', 'date_code', 'summary_date']], 
            on='summary_uid', 
            how='inner'
        )

        print(f"judgments_df after merging: {judgments_df}", type(judgments_df))
        print(f"types of columns in judgments_df: {judgments_df.dtypes}")

        if date_code is not None:
            judgments_df = judgments_df[judgments_df['date_code'] == date_code]
            
    else:
        logger.warning(
            f"Summary file {summaries_jsonl_path} not found, cannot get "
            f"summary_date"
        )

    # TODO: Give readable names?
    # judgment_df_has_data = len(judgments_df) > 0
    # version_mismatch = hhem_version != judgments_df[BasicJudgment.Keys.HHEM_VERSION].iloc[0]
    if (
        len(judgments_df) > 0 
        and hhem_version != judgments_df[BasicJudgment.Keys.HHEM_VERSION].iloc[0]
    ):
        logger.warning(
            f"HHEM version mismatch between passed-in hhem_version and "
            "hhem_version in judgments_df loaded from {judgments_jsonl_path}"
        )
    elif len(judgments_df) == 0:
        logger.warning(
            f"No judgments found after filtering by date_code {date_code} "
            f"in {judgments_jsonl_path}"
        )

    grouped_judgments_df = judgments_df.groupby(
        ['summary_date', 'judgment_date']
    )
    
    for (
        (summary_date, judgment_date),
        subset_df
    ) in tqdm(
        grouped_judgments_df,
        total=len(grouped_judgments_df),
        desc="Date Group Loop"
    ):
        hr = round(compute_hallucination_rate(subset_df)*100.0, 1)
        ar = round(compute_answer_rate(subset_df)*100.0, 1)
        awc = round(compute_avg_word_count(subset_df), 1)
        ci = round(compute_confidence_interval(subset_df)*100.0, 1)

        result_record = Stats(
            eval_name=eval_config.eval_name,
            summary_date=summary_date,
            judgment_date=judgment_date,
            hhem_version=hhem_version,
            **llm_config.model_dump(exclude_none=True), # FIXME: Rethink whether we should exclude_none, exclude_default, or more
            hallucination_rate=hr,
            confidence_interval=ci,
            answer_rate=ar,
            avg_word_count=awc
        )
        
        append_record_to_jsonl(stats_jsonl_path, result_record)