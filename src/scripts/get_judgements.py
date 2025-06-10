from src.logging.Logger import logger
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timezone
import os
from tqdm import tqdm
from src.utils.json_utils import save_to_jsonl, json_exists
from src.analytics.metrics import is_valid_summary
from src.data_struct.data_model import Judgement, Summary, SourceArticle

from src.HHEM.HHEM_2_x import HHEM_2_3, HHEMOutput

from src.LLMs.AbstractLLM import AbstractLLM
from src.scripts.get_summaries import SUMMARY_FILE


#TODO: Documentation Update
"""
For all LLMs in the input list, check if they have a summary file. If it has a 
summary file then it will produce a variety of metrics per summary. Most notably
it produces the HHEM score metric

Functions:
    run(models, article_df, force)
    run_metrics_save_flow(
        hhem_model, article_summary_df, judge_jsonl_path, model_name, force
    )
    calc_and_save_metrics(hhem_model, article_summary_df, judge_jsonl_path)
    build_metric_records(
        article_ids, article_summaries, hhem_scores, hhem_version
    )

Global Variables
    JUDGEMENT_FILE
"""

JUDGEMENT_FILE = "judgements.jsonl"

def run(models: list[AbstractLLM], article_df: pd.DataFrame, force: bool):
    """
    For the given model lists, checks if they have valid summaries.jsonl then 
    calcs metrics for each summary, builds summary objects, then saves them to jsonl

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
            article_summary_df = pd.merge(
                article_df, summaries_df,
                on=Summary.Keys.ARTICLE_ID, how='inner'
            )

            judgements_jsonl_file = f"{JUDGEMENT_FILE}"
            judgements_jsonl_path = os.path.join(
                model_out_dir, judgements_jsonl_file
            )
            run_metric_save_flow(
                hhem_model,
                article_summary_df,
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

def run_metric_save_flow(
        hhem_model: HHEM_2_3,
        article_summary_df: pd.DataFrame,
        judge_jsonl_path: str,
        model_name: str,
        force: bool
    ):
    """
    Controls logic flow for calculating and saving metrics, only produces 
    metrics if a judgments.jsonl file dos not exists unless force flag enabled.

    Args:
        hhem_model (HHEM_2_3): hhem model
        article_summary_df (pd.DataFrame): data containing source articles and 
            summaries aligned
        judge_jsonl_path (str): path for new or possibly existing jsonl file
        model_name (str): name of model that generated the summaries
        force (bool): flag that forces file to be overwritten even if it exists
    """

    if json_exists(judge_jsonl_path) and not force:
        print((
            f"WARNING: {JUDGEMENT_FILE} file already exists, "
            "if you generated new "
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
        calc_and_save_metrics(
            hhem_model, article_summary_df, judge_jsonl_path
        )
        logger.log(f"Finished generating and saving {JUDGEMENT_FILE}")
        logger.log("Moving on to next model")


def calc_and_save_metrics(
        hhem_model: HHEM_2_3,
        article_summary_df: pd.DataFrame,
        judge_jsonl_path: str
    ):
    #TODO: Refactor this function and build metric records
    """
    Calculates the HHEM score, builds metric records, then saves

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
    metric_records = []

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
        metric_records.append(metric_record)

    save_to_jsonl(judge_jsonl_path, metric_records)

    # hhem_scores = []
    # hhem_labels = []
    # for premise, hypothesis in tqdm(zip(article_texts, article_summaries), total=len(article_texts), desc="HHEM Loop"):
    #     input = (premise, hypothesis)
    #     hhem_out = hhem_model.predict(*input)
    #     hhem_scores.append(hhem_out.score)
    #     hhem_labels.append(hhem_out.label)
    # metric_records = build_metric_records(
    #     article_ids, article_summaries, hhem_scores, hhem_model.__str__()
    # )

# UNUSED FUNCTION ATM
def build_metric_records(
        article_ids: list[int], article_summaries: list[str],
        hhem_scores: list[float], hhem_version: str
    ) -> list[Judgement]:
    #TODO: Refactor this faction and calc and save metrics
    """
    For each entry calculates some extra metrics then builds a Judgement object.

    Args:
        article_ids (list[int]):
        article_summaries (list[str])
        hhem_scores (list[float]):
        hhem_model_name (str):
    Returns:
        list[Judgement]: list of Judgement objects
    """
    metric_records = []
    current_date = datetime.now(timezone.utc).date().isoformat()
    for a_id, summ, hhem_s in zip(article_ids, article_summaries, hhem_scores):
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