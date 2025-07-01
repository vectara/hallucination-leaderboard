from src.Logger import logger
import os
from tqdm import tqdm
from src.LLMs.AbstractLLM import AbstractLLM
from src.utils.json_utils import file_exists, append_record_to_jsonl
import pandas as pd
from datetime import datetime, timezone
from src.data_struct.data_model import Stats
from src.analytics.stats import (
    compute_hallucination_rate, compute_answer_rate,
    compute_avg_summary_length, compute_confidence_interval
)

from src.scripts.get_judgements import JUDGEMENT_FILE

"""
Computes and saves statistics for all given models

Global Variables:
    RESULTS_FILE

Functions:
    run(models)
    generate_and_save_results(model_name, judge_jsonl_path, results_jsonl_path)
"""

RESULTS_FILE = "stats.jsonl"

def run(models: list[AbstractLLM]):
    """
    Verifies judgement file exists then computes and saves final results

    Args:
        models (list[AbstractLLM]): list of llms

    Returns:
        None
    """
    logger.info("Starting results computation")

    for model in tqdm(models, desc="Model Loop"):
        model_name = model.get_model_name()
        model_out_dir = model.get_model_out_dir()

        logger.info(f"Generating results for {model_name}")

        judge_jsonl_path = os.path.join(model_out_dir, JUDGEMENT_FILE)

        if file_exists(judge_jsonl_path):
            logger.info(f"{JUDGEMENT_FILE} found for {model_name}")

            results_jsonl_file = f"{RESULTS_FILE}"
            results_jsonl_path = os.path.join(model_out_dir, results_jsonl_file)
            open(results_jsonl_path, 'w').close()
            generate_and_save_results(
                model_name, judge_jsonl_path, results_jsonl_path
            )
        else:
            logger.warning(
                f"{JUDGEMENT_FILE} not found for {model_name}, skipping model"
            )
    logger.info("Finished generating and saving results for all models")

def generate_and_save_results(
        model_name: str, judge_jsonl_path: str, results_jsonl_path: str
    ):
    """
    Loads metrics, computes all stats, formats them, and saves as jsonl file.
    Date codes are grouped and then stats are performed on them seperately.
    Each date code has its own entry in the output file.

    Args:
        model_name (str): name of model
        judge_jsonl_path (str): path to metrics jsonl
        results_jsonl_path (str): path to new json file

    Returns:
        None
    """
    results = {}
    current_date = datetime.now(timezone.utc).date().isoformat()
    metrics_df = pd.read_json(judge_jsonl_path, lines=True)
    grouped_metric_df = metrics_df.groupby(Stats.Keys.DATE_CODE)

    for date_code, subset_df in tqdm(grouped_metric_df, total=len(grouped_metric_df), desc="Date Code Loop"):

        hr = round(compute_hallucination_rate(subset_df)*100.0, 1)
        ar = round(compute_answer_rate(subset_df)*100.0, 1)
        asl = round(compute_avg_summary_length(subset_df), 1)
        ci = round(compute_confidence_interval(subset_df)*100.0, 1)

        result_record = Stats(
            timestamp=current_date,
            llm=model_name,
            date_code=str(date_code),
            hallucination_rate=hr,
            confidence_interval=ci,
            answer_rate=ar,
            avg_summary_length=asl
        )

        append_record_to_jsonl(results_jsonl_path, result_record)