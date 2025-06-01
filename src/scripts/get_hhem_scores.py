from src.logging.Logger import logger
from typing import Literal
from tqdm import tqdm
import pandas as pd
import inspect
import os
from tqdm import tqdm
from src.utils.json_utils import save_to_json, json_exists, load_json

from src.HHEM.HHEM_2_x import HHEM_2_3, HHEMOutput

from src.LLMs.AbstractLLM import AbstractLLM

"""
Gets the HHEM scores for all LLMs that have an existing summary JSON file. HHEM
score data is stored as a JSON file local to the associated LLM class.

Functions:
    run(models)
    generate_and_save_hhem_scores(hhem_model, df, hhem_json_path)
    create_hhem_records(article_ids, hhem_scores, hhem_labels)
"""

def run(models: list[AbstractLLM], article_df: pd.DataFrame, force: bool):
    """
    Generates and saves HHEM scores for a given model only if it has its 
    respective summaries_model.json file

    Args:
        models (list[AbstractLLM]): list of LLMs
        force (bool): flag for forcing json to be overwritten if it exists
    Returns:
        None
    """
    logger.log("Starting to generate HHEM scores")
    if force:
        logger.log("Force flag enabled. Overwriting previous JSON data")

    hhem_model = HHEM_2_3()

    for model in tqdm(models, desc="Model Loop"):
        model_name = model.get_model_name()
        model_out_dir = model.get_model_out_dir()

        logger.log(f"Generating HHEM scores for {model_name}")

        summaries_json_file = f"summaries_{model_name}.json"
        summaries_json_path = os.path.join(model_out_dir, summaries_json_file)

        if json_exists(summaries_json_path):
            logger.log(f"Summary JSON found for {model_name}")
            summaries_json = load_json(summaries_json_path)
            summaries_df = pd.DataFrame(summaries_json["summaries"])
            article_summaries_df = pd.merge(
                article_df, summaries_df,
                on='article_id', how='inner'
            )

            hhem_json_file = f"hhem_scores_{model_name}.json"
            hhem_json_path = os.path.join(model_out_dir, hhem_json_file)
            run_generation_save_flow(
                hhem_model,
                article_summaries_df,
                hhem_json_path,
                model_name,
                force
            )
        else:
            logger.log(
                f"Summary JSON not found for {model_name}, skipping model"
            )
    logger.log("Finished generating and saving HHEM scores for all models")

def run_generation_save_flow(
        hhem_model: HHEM_2_3,
        df: pd.DataFrame,
        hhem_json_path: str,
        model_name: str,
        force: bool
    ):
    """
    Controls logic flow for generating and saving HHEM scores depending on
    force tag and whether JSON files exist

    hhem_model (HHEM_2_3): hhem model
    df (pd.DataFrame): data containing source articles and summaries aligned
    hhem_json_path (str): path for new or possibly existing JSON file
    model_name (str): name of model that generated the summaries
    force (bool): flag that forces file to be overwritten even if it exists
    """
    if json_exists(hhem_json_path) and not force:
        print((
            "WARNING: HHEM JSON file already exists, if you generated new "
            "summaries you will not have HHEM scores that reflect these "
            "summaries. Recall with --force to overwrite old data"
            )
        )
        logger.log(f"HHEM JSON file exists for {model_name}, skipping")
    else:
        if not force:
            logger.log("HHEM JSON file does not exist, generating...")
        else:
            logger.log("Overwriting previous HHEM score JSON...")
        generate_and_save_hhem_scores(
            hhem_model, df, hhem_json_path
        )
        logger.log("Finished generating and saving HHEM scores")
        logger.log("Moving on to next model")


def generate_and_save_hhem_scores(
        hhem_model: HHEM_2_3, df: pd.DataFrame, hhem_json_path: str
    ):
    """
    For a given models output, request the HHEM model to predict the scores and
    save them in a JSON file

    Args:
        hhem_model (HHEM_2_3): HHEM model
        df (DataFrame): contains article and summaries merged on article_id
        hhem_json_path (str): path to new JSON file
    Returns:
        None
    """
    article_texts = df['text'].tolist()
    article_summaries = df['summary'].tolist()
    article_ids = df['article_id'].tolist()

    hhem_scores = []
    hhem_labels = []
    for premise, hypothesis in tqdm(zip(article_texts, article_summaries), total=len(article_texts), desc="HHEM Loop"):
        input = (premise, hypothesis)
        hhem_out = hhem_model.predict(*input)
        hhem_scores.append(hhem_out.score)
        hhem_labels.append(hhem_out.label)
    hhem_records = create_hhem_records(article_ids, hhem_scores, hhem_labels)
    save_to_json(hhem_json_path, hhem_records)

def create_hhem_records(
        article_ids: list[int],
        hhem_scores: list[float], hhem_labels: list[Literal[0,1]]
    ):
    """
    Creates the HHEM score records for a given article_id

    Current JSON format, *Format may not align with code in future, check code*
    {
        'article_id': int
        'hhem_score': float
        'hhem_label': Literal[0,1]
    }

    Args:
        article_ids (list[int]):
        hhem_scores (list[float]):
        hhem_labels (list[Literal[0,1]]):
    Returns:
        list[dict]: hhem score records in JSON format
    """
    hhem_score_records = [
        {
            "article_id": a_id,
            "hhem_score": hhem_s,
            "hhem_label": hhem_l
        }
        for a_id, hhem_s, hhem_l in zip(
            article_ids, hhem_scores, hhem_labels
        )
    ]
    return hhem_score_records

if __name__ == "__main__":
    pass