from src.logging.Logger import logger
import os
from tqdm import tqdm
from src.LLMs.AbstractLLM import AbstractLLM
from src.utils.json_utils import json_exists, save_to_json
from src.metrics.HHEMMetrics import HHEMMetrics
import pandas as pd


def run(models: list[AbstractLLM]):
    logger.log("Starting results computation")

    for model in tqdm(models, desc="Model Loop"):
        model_name = model.get_model_name()
        model_out_dir = model.get_model_out_dir()

        logger.log(f"Generating results for {model_name}")

        summaries_json_file = f"summaries_{model_name}.json"
        summaries_json_path = os.path.join(model_out_dir, summaries_json_file)
        hhem_json_file = f"hhem_scores_{model_name}.json"
        hhem_json_path = os.path.join(model_out_dir, hhem_json_file)

        if json_exists(summaries_json_path) and json_exists(hhem_json_path):
            logger.log(f"Summary and HHEM JSON found for {model_name}")
            summaries_df = pd.read_json(summaries_json_path)
            hhem_df = pd.read_json(hhem_json_path)
            if len(hhem_df) != len(summaries_df):
                logger.log("HHEM Summaries data length mismatch, skipping model")
                continue
            hhem_summaries_df = pd.merge(
                hhem_df, summaries_df,
                on='article_id', how='inner'
            )

            results_json_file = f"results_{model_name}.json"
            results_json_path = os.path.join(model_out_dir, results_json_file)
            generate_and_save_results(hhem_summaries_df, results_json_path)
        else:
            logger.log(
                f"Summary or HHEM JSON not found for {model_name}, skipping model"
            )
    logger.log("Finished generating and saving results for all models")

def generate_and_save_results(df: pd.DataFrame, results_json_path: str):
    """

    Args:
        df (DataFrame): contains hhem and summaries merged on article_id
        results_json_path (str): path to new JSON file
    Returns:
        None
    """
    article_summaries = df['summary'].tolist()
    article_hhem_scores = df['hhem_score'].tolist()
    results = {}
    metrics = HHEMMetrics()

    hr = round(
        metrics.compute_hallucination_rate(article_hhem_scores, article_summaries)*100.0, 1
    )
    fcr = round(
        metrics.compute_factual_consistancy_rate(article_hhem_scores, article_summaries)*100.0, 1
    )
    ar = round(metrics.compute_answer_rate(article_summaries)*100.0, 1)
    asl = round(metrics.compute_avg_summary_length(article_summaries), 1)

    results["hallucination_rate"] = hr
    results["factual_consistancy_rate"] = fcr 
    results["answer_rate"] = ar
    results["average_summary_length"] = asl

    save_to_json(results_json_path, results)