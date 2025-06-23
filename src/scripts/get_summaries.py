from src.logging.Logger import logger
import pandas as pd
import os
from tqdm import tqdm
from src.utils.json_utils import file_exists, append_record_to_jsonl
from datetime import datetime, timezone
from src.data_struct.data_model import Summary, SourceArticle
import hashlib

from src.LLMs.AbstractLLM import AbstractLLM

"""
Loops through all given LLMs and requests a summary for the give article
dataframe. 

Global Variables:
    SUMMARY_FILE

Functions:
    run(models)
    generate_and_save_summaries(model, article_df, json_path)
"""

SUMMARY_FILE = "summaries.jsonl"

def run(models: list[AbstractLLM], article_df: pd.DataFrame, ow=False):
    """
    Generates summaries for a given model if the corresponding jsonl file does 
    not exist, overwrite flag will overwrite existing jsonl file

    Args:
        models (list[AbstractLLM]): list of LLMs
        article_df (pd.DataFrame): article dataset
        ow (bool): flag to specify if JSON should still be created if exists

    Returns:
        None
    """

    logger.log(f"Starting to generate {SUMMARY_FILE}")
    if ow:
        logger.log(
            f"Overwrite flag enabled"
        )

    for model in tqdm(models, desc="Model Loop"):
        model_name = model.get_model_name()
        model_out_dir = model.get_model_out_dir()

        logger.log(f"Generating {SUMMARY_FILE} for {model_name}")

        jsonl_file = f"{SUMMARY_FILE}"
        summaries_jsonl_path = os.path.join(model_out_dir, jsonl_file)

        if not file_exists(summaries_jsonl_path):
            logger.log(f"{SUMMARY_FILE} file does not exist, generating...")
            open(summaries_jsonl_path, 'w').close()
        elif file_exists(summaries_jsonl_path) and ow:
            logger.log(f"Overwriting previous data in {SUMMARY_FILE}")
            open(summaries_jsonl_path, 'w').close()
        else:
            logger.log(f"Adding additional data to {SUMMARY_FILE}")

        generate_and_save_summaries(model, article_df, summaries_jsonl_path)
        logger.log(f"Finished generating and saving for {model_name}")
        
        logger.log("Moving on to next model")
    
    logger.log(f"Finished generating and saving {SUMMARY_FILE} for all models")

def generate_and_save_summaries(
        model: AbstractLLM,
        article_df: pd.DataFrame,
        jsonl_path: str
    ):
    #TODO: Update doc
    """
    Produces summaries for all articles and saves them to the given jsonl file.
    Saving is performed incrementally.

    Args:
        model (AbstractLLM): LLM model
        article_df (pd.DataFrame): Article data
        jsonl_path (str): path for the new jsonl file

    Returns:
        None
    """

    current_date = datetime.now(timezone.utc).date().isoformat()
    article_texts = article_df[SourceArticle.Keys.TEXT].tolist()
    article_ids = article_df[SourceArticle.Keys.ARTICLE_ID].tolist()

    logger.log("Appending to jsonl file")
    with model as m: 
        for article, a_id in tqdm(
            zip(article_texts, article_ids),
            total=len(article_texts),
            desc="Article Loop"
        ):
            summary = m.summarize_clean_wait(article)
            summary_uid = generate_summary_uid(
                model.get_model_name(),
                model.get_date_code(),
                summary,
                current_date
            )
            record = Summary(
                timestamp=current_date,
                summary_uid=summary_uid,
                llm=model.get_model_name(),
                date_code=model.get_date_code(),
                article_id=a_id,
                summary=summary
            )
            append_record_to_jsonl(jsonl_path, record)


def generate_summary_uid(model, date_code, summary_text, date):
    #TODO: Docs
    """
    """
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    combined_string = f"{model}|{date_code}|{summary_text.strip()}|{date}|{current_time}"
    return hashlib.md5(combined_string.encode('utf-8')).hexdigest()

            
if __name__ == "__main__":
    pass