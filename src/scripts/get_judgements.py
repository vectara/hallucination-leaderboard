from src.logging.Logger import logger
from typing import Literal
from tqdm import tqdm
import pandas as pd
import inspect
from datetime import datetime, timezone
import os
from tqdm import tqdm
from src.utils.json_utils import save_to_jsonl, json_exists
from src.metrics.metrics import is_valid_summary
from src.data_struct.data_model import Judgement, Summary, SourceArticle

from src.HHEM.HHEM_2_x import HHEM_2_3, HHEMOutput

from src.LLMs.AbstractLLM import AbstractLLM
from src.scripts.get_summaries import SUMMARY_FILE


#TODO: Documentation Update
"""
Gets the metrics for all LLMs that have an existing summary JSONL file

Functions:
    run(models)
    generate_and_save_hhem_scores(hhem_model, df, hhem_json_path)
    create_hhem_records(article_ids, hhem_scores, hhem_labels)
"""

# METRICS_FILE_PREFIX = "judgements"
# OUTPUT_FILE_TYPE = "jsonl"
JUDGEMENT_FILE = "judgements.jsonl"

def run(models: list[AbstractLLM], article_df: pd.DataFrame, force: bool):
    #TODO: Documentation Update, style
    """
    Generates and saves HHEM scores for a given model only if it has its 
    respective summaries_model.json file

    Args:
        models (list[AbstractLLM]): list of LLMs
        article_df (pd.DataFrame): article dataset
        force (bool): flag for forcing json to be overwritten if it exists
    Returns:
        None
    """
    logger.log(f"Starting to generate {JUDGEMENT_FILE} scores")
    if force:
        logger.log(
            f"Force flag enabled. Overwriting previous {JUDGEMENT_FILE} data"
        )

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
            article_summaries_df = pd.merge(
                article_df, summaries_df,
                on=Summary.Keys.ARTICLE_ID, how='inner'
            )

            judgements_jsonl_file = f"{JUDGEMENT_FILE}"
            judgements_jsonl_path = os.path.join(
                model_out_dir, judgements_jsonl_file
            )
            run_generation_save_flow(
                hhem_model,
                article_summaries_df,
                judgements_jsonl_path,
                model_name,
                force
            )
        else:
            logger.log(
                f"{SUMMARY_FILE} not found for {model_name}, skipping model"
            )
    logger.log(
        f"Finished generating and saving {JUDGEMENT_FILE} for all models"
    )

def run_generation_save_flow(
        hhem_model: HHEM_2_3,
        df: pd.DataFrame,
        judge_jsonl_path: str,
        model_name: str,
        force: bool
    ):
    #TODO: Documentation Update
    """
    Controls logic flow for generating and saving HHEM scores depending on
    force tag and whether JSON files exist

    Args:
        hhem_model (HHEM_2_3): hhem model
        df (pd.DataFrame): data containing source articles and summaries aligned
        hhem_json_path (str): path for new or possibly existing JSON file
        model_name (str): name of model that generated the summaries
        force (bool): flag that forces file to be overwritten even if it exists
    """

    if json_exists(judge_jsonl_path) and not force:
        print((
            f"WARNING: {JUDGEMENT_FILE} file already exists, if you generated new "
            "summaries you will not have metrics that reflect these "
            "summaries. Recall with --force to overwrite old data"
            )
        )
        logger.log(f"{JUDGEMENT_FILE} file exists for {model_name}, skipping")
    else:
        if not force:
            logger.log(f"{JUDGEMENT_FILE} file does not exist, generating...")
        else:
            logger.log(f"Overwriting previous {JUDGEMENT_FILE}...")
        generate_and_save_metrics(
            hhem_model, df, judge_jsonl_path
        )
        logger.log(f"Finished generating and saving {JUDGEMENT_FILE}")
        logger.log("Moving on to next model")


def generate_and_save_metrics(
        hhem_model: HHEM_2_3, df: pd.DataFrame, judge_jsonl_path: str
    ):
    #TODO: Documentation Update
    """
    For a given models output, request the HHEM model to predict the scores and
    save them in a JSON file

    Args:
        hhem_model (HHEM_2_3): HHEM model
        df (DataFrame): contains article and summaries merged on article_id
        judge_jsonl_path (str): path to new JSON file
    Returns:
        None
    """
    article_texts = df[SourceArticle.Keys.TEXT].tolist()
    article_summaries = df[Summary.Keys.SUMMARY].tolist()
    article_ids = df[Summary.Keys.ARTICLE_ID].tolist()

    hhem_scores = []
    hhem_labels = []
    for premise, hypothesis in tqdm(zip(article_texts, article_summaries), total=len(article_texts), desc="HHEM Loop"):
        input = (premise, hypothesis)
        hhem_out = hhem_model.predict(*input)
        hhem_scores.append(hhem_out.score)
        hhem_labels.append(hhem_out.label)
    metric_records = create_metric_records(
        article_ids, article_summaries, hhem_scores, hhem_labels, hhem_model.__str__()
    )
    save_to_jsonl(judge_jsonl_path, metric_records)

def create_metric_records(
        article_ids: list[int], article_summaries: list[str],
        hhem_scores: list[float], hhem_labels: list[Literal[0,1]],
        hhem_version: str
    ):
    #TODO: Update Documetation
    """
    Creates the HHEM score records for a given article_id

    Current JSON format, *Format may not align with code in future, check code*
    {
        'article_id': int
        'hhem_score': float
        'hhem_label': Literal[0,1]
        'summary_length': int
        'valid_summary': bool
    }

    Args:
        article_ids (list[int]):
        article_summaries (list[str])
        hhem_scores (list[float]):
        hhem_labels (list[Literal[0,1]]):
        hhem_model_name (str):
    Returns:
        list[dict]: hhem score records in JSON format
    """
    metric_records = []
    current_date = datetime.now(timezone.utc).date().isoformat()
    for a_id, summ, hhem_s, hhem_l in zip(article_ids, article_summaries, hhem_scores, hhem_labels):
        summary_length = len(summ.split())
        valid_summary = is_valid_summary(summ)
        metric_record = Judgement(
            timestamp = current_date,
            article_id = a_id,
            hhem_version = hhem_version,
            hhem_score = hhem_s,
            valid=valid_summary,
            summary_words=summary_length
        )
        metric_records.append(metric_record)

    return metric_records

if __name__ == "__main__":
    pass