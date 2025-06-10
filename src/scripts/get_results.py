from src.logging.Logger import logger
import os
from tqdm import tqdm
from src.LLMs.AbstractLLM import AbstractLLM
from src.utils.json_utils import json_exists, save_bm_to_json
import pandas as pd
from datetime import datetime, timezone
from src.data_struct.data_model import Stats
from src.analytics.stats import (
    compute_hallucination_rate, compute_answer_rate, compute_avg_summary_length
)

from src.scripts.get_judgements import JUDGEMENT_FILE


"""
Computes and saves results for list of all models

Functions:
    run(models)
    generate_and_save_results(df, results_json_path)
"""
RESULTS_FILE = "stats.json"

def run(models: list[AbstractLLM]):
    #TODO: Update Documentation
    """
    For all models setup the necessary data needed to compute and save results

    Args:
        models (list[AbstractLLM]): list of llms

    Returns:
        None
    """
    logger.log("Starting results computation")

    for model in tqdm(models, desc="Model Loop"):
        model_name = model.get_model_name()
        model_out_dir = model.get_model_out_dir()

        logger.log(f"Generating results for {model_name}")

        judge_json_path = os.path.join(model_out_dir, JUDGEMENT_FILE)

        if json_exists(judge_json_path):
            logger.log(f"{JUDGEMENT_FILE} found for {model_name}")

            results_json_file = f"{RESULTS_FILE}"
            results_json_path = os.path.join(model_out_dir, results_json_file)
            generate_and_save_results(judge_json_path, model_name, results_json_path)
        else:
            logger.log(
                f"{JUDGEMENT_FILE} not found for {model_name}, skipping model"
            )
    logger.log("Finished generating and saving results for all models")

def generate_and_save_results(judge_json_path: str, model_name: str, results_json_path: str):
    #TODO: Update Documentation
    """
    Loads metrics, computes all stats, formats them, and saves them to disk as JSON file

    Args:
        hhem_json_path (str): path to metrics JSON
        model_name (str): name of model
        results_json_path (str): path to new JSON file
    Returns:
        None
    """
    results = {}
    current_date = datetime.now(timezone.utc).date().isoformat()

    metrics_df = pd.read_json(judge_json_path, lines=True)

    hr = round(compute_hallucination_rate(metrics_df)*100.0, 1)
    ar = round(compute_answer_rate(metrics_df)*100.0, 1)
    asl = round(compute_avg_summary_length(metrics_df), 1)



    results = Stats(
        timestamp=current_date,
        llm=model_name,
        hallucination_rate=hr,
        answer_rate=ar,
        avg_summary_length=asl
    )

    save_bm_to_json(results_json_path, results)