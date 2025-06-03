from src.logging.Logger import logger
import pandas as pd
import inspect
import os
from tqdm import tqdm
from src.utils.json_utils import save_to_json, json_exists
from datetime import datetime, timezone
from src.metrics.HHEMMetrics import HHEMMetrics

from src.LLMs.AbstractLLM import AbstractLLM

"""
Requests all LLMS to produce a summary. Summaries are only produced if no 
summary data is detected. Summaries can be forced to be regenerated. Stores the
summary data as a JSON file local to associated LLM class.

Functions:
    run(models)
    generate_and_save_summaries(model, article_df, json_path)
    create_summary_records(summaries, article_df)
"""

# TODO: Add Valid summary field

def run(models: list[AbstractLLM], article_df: pd.DataFrame, force=False):
    """
    Generates summaries for a given model if the corresponding JSON file does 
    not exist

    Args:
        models (list[AbstractLLM]): list of LLMs
        article_df (pd.DataFrame): article dataset
        force (bool): flag to specify if JSON should still be created if exists

    Returns:
        None
    """

    logger.log("Starting to generate summaries")
    if force:
        logger.log("Force flag enabled. Overwriting previous JSON data")

    for model in tqdm(models, desc="Model Loop"):
        model_name = model.get_model_name()
        model_out_dir = model.get_model_out_dir()

        logger.log(f"Generating summaries for {model_name}")

        json_file = f"summaries_{model_name}.json"
        summaries_json_path = os.path.join(model_out_dir, json_file)

        if json_exists(summaries_json_path) and not force:
            logger.log(f"Summaries JSON file exists for {model_name}, skipping")
            continue
        else:
            if not force:
                logger.log("Summaries JSON file does not exist, generating...")
            generate_and_save_summaries(model, article_df, summaries_json_path)
            logger.log(f"Finished generating and saving for {model_name}")
        
        logger.log("Moving on to next model")
    
    logger.log("Finished generating and saving summaries for all models")

def generate_and_save_summaries(
        model: AbstractLLM,
        article_df: pd.DataFrame,
        json_path: str
    ):
    """
    Generates the summaries, reformats the data for a JSON file, and saves the
    record to a JSON file in the folder with the corresponding object.

    Args:
        model (AbstractLLM): LLM model
        article_df (pd.DataFrame): Article data
        json_path (str): path for the new json file

    Returns:
        None
    """

    article_texts = article_df['text'].tolist()
    article_ids = article_df['article_id'].tolist()
    summaries = []
    with model as m: 
        summaries = m.summarize_articles(article_texts)
    summary_records = create_summary_records(summaries, article_ids)
    save_to_json(json_path, summary_records)

def create_summary_records(
        summaries: list[str],
        article_ids: list[int]
    ) -> list[dict]:
    """
    Reformats summary and article data into JSON format

    Current JSON format, *Format may not align with code in future, check code*
    {
        'article_id': int
        'summary': str
    }

    Args:
        summaries (list[str]): List of summaries
        article_ids (list[int]): id associated with an article
    
    Returns:
        (list): JSON formatted dictionary
    """
    model_summaries = []
    metrics = HHEMMetrics()

    for a_id, summ in zip(article_ids, summaries):
        record = {
            "article_id": a_id,
            "summary": summ,
            "valid_summary": metrics.is_valid_summary(summ)
        }
        model_summaries.append(record)


    current_utc_time = datetime.now(timezone.utc).isoformat()

    package = {
        "timestamp": current_utc_time,
        "summaries": model_summaries
    }
    return package

if __name__ == "__main__":
    pass