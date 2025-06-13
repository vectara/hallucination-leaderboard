from src.logging.Logger import logger
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timezone
import os
from tqdm import tqdm
from src.utils.json_utils import json_exists, append_record_to_jsonl
from src.analytics.metrics import is_valid_summary
from src.data_struct.data_model import Judgement, Summary, SourceArticle

from src.HHEM.HHEM_2_x import HHEM_2_3, HHEMOutput

from src.LLMs.AbstractLLM import AbstractLLM
from src.scripts.get_summaries import SUMMARY_FILE


#TODO: Documentation Update
"""
For all LLMs in the input list, check if they have a summary file. If it has a 
summary file then it will produce a variety of metrics per summary. Most notably
it produces the HHEM score metric.

Global Variables:
    JUDGEMENT_FILE

Functions:
    run(models, article_df)
    calc_and_save_metrics(hhem_model, article_summary_df, judge_jsonl_path)
"""

JUDGEMENT_FILE = "judgements.jsonl"

def run(models: list[AbstractLLM], article_df: pd.DataFrame):
    """
    For the given model lists, checks if they have valid summaries.jsonl then 
    calcs metrics for each summary, builds summary objects, then saves them
    to jsonl

    Args:
        models (list[AbstractLLM]): list of LLMs
        article_df (pd.DataFrame): article dataset
    Returns:
        None
    """
    logger.log(f"Starting to generate {JUDGEMENT_FILE} scores")

    hhem_model = HHEM_2_3()

    for model in tqdm(models, desc="Model Loop"):
        model_name = model.get_model_name()
        model_out_dir = model.get_model_out_dir()

        logger.log(f"Generating {JUDGEMENT_FILE} scores for {model_name}")

        summaries_jsonl_file = f"{SUMMARY_FILE}"
        summaries_jsonl_path = os.path.join(model_out_dir, summaries_jsonl_file)

        if json_exists(summaries_jsonl_path):
            logger.log(f"{SUMMARY_FILE} found for {model_name}")
            summaries_df = pd.read_json(summaries_jsonl_path, lines=True)
            article_summary_df = pd.merge(
                article_df, summaries_df,
                on=Summary.Keys.ARTICLE_ID, how='inner'
            )

            judgements_jsonl_file = f"{JUDGEMENT_FILE}"
            judgements_jsonl_path = os.path.join(
                model_out_dir, judgements_jsonl_file
            )
            open(judgements_jsonl_path, 'w').close()

            calc_and_save_metrics(
                hhem_model, article_summary_df, judgements_jsonl_path
            )
        else:
            logger.log(
                f"{SUMMARY_FILE} not found for {model_name}, skipping model"
            )
    logger.log(
        f"Finished generating and saving {JUDGEMENT_FILE} for all models"
    )

def calc_and_save_metrics(
        hhem_model: HHEM_2_3,
        article_summary_df: pd.DataFrame,
        judge_jsonl_path: str
    ):
    """
    Produces metrics for Articles and Summaires aligned on article_id

    Output is incrementally saved to the given jsonl file

    Args:
        hhem_model (HHEM_2_3): HHEM model
        article_summary_df (DataFrame): contains article and summaries merged 
            on article_id
        judge_jsonl_path (str): path to new jsonl file
    Returns:
        None
    """
    article_texts = article_summary_df[SourceArticle.Keys.TEXT].tolist()
    article_summaries = article_summary_df[Summary.Keys.SUMMARY].tolist()
    article_ids = article_summary_df[Summary.Keys.ARTICLE_ID].tolist()

    current_date = datetime.now(timezone.utc).date().isoformat()

    for premise, hypothesis, a_id in tqdm(
        zip(article_texts, article_summaries, article_ids),
        total=len(article_texts),
        desc="HHEM Loop"
    ):
        input = (premise, hypothesis)
        hhem_out = hhem_model.predict(*input)
        summary_length = len(hypothesis.split())
        valid_summary = is_valid_summary(hypothesis)
        metric_record = Judgement(
            timestamp = current_date,
            article_id = a_id,
            hhem_version = hhem_model.__str__(),
            hhem_score = hhem_out.score,
            valid=valid_summary,
            summary_words=summary_length
        )
        append_record_to_jsonl(judge_jsonl_path, metric_record)

if __name__ == "__main__":
    pass