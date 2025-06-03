from src.logging.Logger import logger
import os
from tqdm import tqdm
from src.LLMs.AbstractLLM import AbstractLLM
from src.utils.json_utils import json_exists, save_to_json, load_json
from src.metrics.HHEMMetrics import HHEMMetrics
import pandas as pd
from datetime import datetime, timezone


"""
Computes and saves results for list of all models

Functions:
    run(models)
    generate_and_save_results(df, results_json_path)
"""

def run(models: list[AbstractLLM]):
    """
    UPDATE
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

        # summaries_json_file = f"summaries_{model_name}.json"
        # summaries_json_path = os.path.join(model_out_dir, summaries_json_file)
        hhem_json_file = f"hhem_scores_{model_name}.json"
        hhem_json_path = os.path.join(model_out_dir, hhem_json_file)

        # if json_exists(summaries_json_path) and json_exists(hhem_json_path):
        if json_exists(hhem_json_path):
            logger.log(f"Summary and HHEM JSON found for {model_name}")
            # summaries_json = load_json(summaries_json_path)
            # summaries_df = pd.DataFrame(summaries_json["summaries"])
            # hhem_json = load_json(hhem_json_path)
            # if len(hhem_df) != len(summaries_df):
            #     logger.log("HHEM Summaries data length mismatch, skipping model")
            #     continue
            # hhem_summaries_df = pd.merge(
            #     hhem_df, summaries_df,
            #     on='article_id', how='inner'
            # )

            results_json_file = f"results_{model_name}.json"
            results_json_path = os.path.join(model_out_dir, results_json_file)
            generate_and_save_results(hhem_json_path, model_name, results_json_path)
        else:
            logger.log(
                f"Summary or HHEM JSON not found for {model_name}, skipping model"
            )
    logger.log("Finished generating and saving results for all models")

def generate_and_save_results(hhem_json_path: str, model_name: str, results_json_path: str):
    """
    UPDATE
    Computes all metrics, formats them, and saves them to disk as JSON file

    Args:
        df (DataFrame): contains hhem and summaries merged on article_id
        results_json_path (str): path to new JSON file
    Returns:
        None
    """
    results = {}
    metrics = HHEMMetrics()

    hhem_json = load_json(hhem_json_path)
    hhem_version = hhem_json["hhem_version"]
    metrics_df = pd.DataFrame(hhem_json["hhem_scores"])

    hr = round(
        metrics.compute_hallucination_rate(metrics_df)*100.0, 1
    )
    ar = round(metrics.compute_answer_rate(metrics_df)*100.0, 1)
    asl = round(metrics.compute_avg_summary_length(metrics_df), 1)

    results["hallucination_rate"] = hr
    results["answer_rate"] = ar
    results["average_summary_length"] = asl

    current_utc_time = datetime.now(timezone.utc).isoformat()

    package = {
        "timestamp": current_utc_time,
        "llm": model_name,
        "hhem_version": hhem_version,
        "results": results
    }

    save_to_json(results_json_path, package)