from src.logging.Logger import logger
import pandas as pd
import inspect
import os
import json

from src.LLMs.AbstractLLM import AbstractLLM
from src.LLMs.OpenAI_GPTd4p1.GPTd4p1 import GPTd4p1

"""
Requests all LLMS to produce a summary. Summaries are only produced if no 
summary data is detected. Summaries can be forced to be regenerated. Stores the
summary data as a JSON file local to associated LLM class.

Functions:
    run()
    generate_and_save_summaries(model, article_df, json_path)
    create_summary_records(summaries, article_df)
    save_to_json(json_path, summary_records)
    summaries_json_exists(full_path)
"""

def run():
    """
    Generates summaries for a given model if the corresponding JSON file does 
    not exist

    Args:
        None

    Returns:
        None
    """
    logger.log("Starting to generate summaries")

    models = [GPTd4p1()]
    article_df = pd.read_csv(os.getenv("LB_DATA"))

    for model in models:
        model_name = model.get_name()

        logger.log(f"Generating summaries for {model_name}")

        obj_file_path = inspect.getfile(type(model))
        obj_dir = os.path.dirname(os.path.abspath(obj_file_path))
        json_file = f"summaries_{model_name}.json"
        json_path = os.path.join(obj_dir, json_file)

        if summaries_json_exists(json_path):
            logger.log(f"Summaries JSON file exists for {model_name}, skipping")
            continue
        else:
            logger.log("Summaries JSON file does not exist, generating...")
            generate_and_save_summaries(model, article_df, json_path)
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
        article_df (DataFrame): Article data
        json_path (str): path for the new json file

    Returns:
        None
    """
    article_texts = article_df['text'].tolist()
    article_ids = article_df['article_id'].tolist()
    summaries = model.summarize_articles(article_texts)
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
    
    Returns:
        (dict): JSON formatted dictionary
    """
    model_summary_dict = [
        {
            "article_id": a_id,
            "summary": summ,
        }
        for a_id, summ in zip(
            article_ids, summaries
        )
    ]
    return model_summary_dict

def save_to_json(json_path: str, summary_records: list[dict]):
    """
    Saves JSON formatted data to disk in the folder with the respective model

    Args:
        json_path (str): Path to the JSON file
        summary_records (list[dict{}]): JSON formatted data

    Returns:
        None
    """
    logger.log("Saving json file")
    with open(json_path, "w") as f:
        json.dump(summary_records, f, indent=4)
    logger.log("JSON file saved")

def summaries_json_exists(full_path: str) -> bool:
    """
    Checks if JSON file exists, returns True if so else False

    Args:
        full_path (str): Path to JSON file

    Returns:
        (bool): State of file existing
    """
    if os.path.isfile(full_path):
        return True
    else:
        return False

if __name__ == "__main__":
    run()