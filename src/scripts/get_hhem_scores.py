from src.logging.Logger import logger
import pandas as pd
import inspect
import os
from src.utils.json_utils import save_to_json, json_exists

from src.HHEM.HHEM_2_x import HHEM_2_3, HHEMOutput

from src.LLMs.AbstractLLM import AbstractLLM

"""
Gets the HHEM scores for all LLMs that have an existing summary JSON file. HHEM
score data is stored as a JSON file local to the associated LLM class.

Functions:

"""

def run(models: list[AbstractLLM]):
    logger.log("Starting to generate HHEM scores")

    article_df = pd.read_csv(os.getenv("LB_DATA"))
    hhem_model = HHEM_2_3()

    for model in models:
        model_name = model.get_name()

        logger.log(f"Generating HHEM scores for {model_name}")

        obj_file_path = inspect.getfile(type(model))
        obj_dir = os.path.dirname(os.path.abspath(obj_file_path))

        summaries_json_file = f"summaries_{model_name}.json"
        summaries_json_path = os.path.join(obj_dir, summaries_json_file)
        if json_exists(summaries_json_path):
            logger.log(f"Summary JSON found for {model_name}")
            summaries_df = pd.read_json(summaries_json_path)
            article_summaries_df = pd.merge(
                article_df, summaries_df,
                on='article_id', how='inner'
            )
            hhem_json_file = f"hhem_scores_{model_name}.json"
            hhem_json_path = os.path.join(obj_dir, hhem_json_file)

            logger.log("Generating HHEM scores...")
            generate_and_save_hhem_scores(
                hhem_model, article_summaries_df, hhem_json_path
            )
            logger.log("Finished generating and saving HHEM scores")
            logger.log("Moving on to next model")
        else:
            logger.log(
                f"Summary JSON not found for {model_name}, skipping model"
            )
            continue
    logger.log("Finished generating and saving HHEM scores for all models")

def generate_and_save_hhem_scores(
        hhem_model: HHEM_2_3, df: pd.DataFrame, hhem_json_path: str
    ):
    article_texts = df['text'].tolist()
    article_summaries = df['summary'].tolist()
    article_ids = df['article_id'].tolist()

    hhem_scores = []
    hhem_labels = []
    for premise, hypothesis in zip(article_texts, article_summaries):
        input = (premise, hypothesis)
        hhem_out = hhem_model.predict(*input)
        hhem_scores.append(hhem_out.score)
        hhem_labels.append(hhem_out.label)
    hhem_records = create_hhem_records(article_ids, hhem_scores, hhem_labels)
    save_to_json(hhem_json_path, hhem_records)

def create_hhem_records(article_ids, hhem_scores, hhem_labels):
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
    run()