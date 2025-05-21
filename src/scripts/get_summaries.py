from src.logging.Logger import logger
import pandas as pd
import inspect
import os
from src.utils.json_utils import save_to_json, json_exists

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

def run(models: list[AbstractLLM]):
    """
    Generates summaries for a given model if the corresponding JSON file does 
    not exist

    Args:
        models (list[AbstractLLM]): list of LLMs

    Returns:
        None
    """
    logger.log("Starting to generate summaries")

    article_df = pd.read_csv(os.getenv("LB_DATA"))

    for model in models:
        model_name = model.get_name()

        logger.log(f"Generating summaries for {model_name}")

        obj_file_path = inspect.getfile(type(model))
        obj_dir = os.path.dirname(os.path.abspath(obj_file_path))
        json_file = f"summaries_{model_name}.json"
        summaries_json_path = os.path.join(obj_dir, json_file)

        if json_exists(summaries_json_path):
            logger.log(f"Summaries JSON file exists for {model_name}, skipping")
            continue
        else:
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
        article_df (DataFrame): Article data
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

if __name__ == "__main__":
    run()