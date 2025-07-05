import os
from datetime import datetime, timezone
from typing import Literal
import pandas as pd
from tqdm import tqdm

from .. analytics import is_valid_summary
from .. data_model import Judgement, SourceArticle, EvalConfig, BasicSummary
from .. HHEM_2_x import HHEM_2_1_open, HHEM_2_3, HHEMOutput
from .. json_utils import append_record_to_jsonl
from .. Logger import logger

def get_hhem_model(hhem_version: Literal["2.1-open", "2.3"]):
    """
    Returns the appropriate HHEM model based on the version string.
    
    Args:
        hhem_version (str): Version string from eval_config
        
    Returns:
        HHEM model instance
    """
    if hhem_version == "2.1-open":
        return HHEM_2_1_open()
    elif hhem_version == "2.3":
        return HHEM_2_3()
    else:
        raise ValueError(f"Unsupported HHEM version: {hhem_version}. Supported versions: 2.1-open, 2.3")

def run(eval_config: EvalConfig, article_df: pd.DataFrame, summary_file: str, judgement_file: str):
    """
    Generate HHEM scores for each summary

    """
    logger.info(f"Starting to generate {judgement_file} scores")

    hhem_model = get_hhem_model(eval_config.hhem_version)

    for llm_config in tqdm(eval_config.LLM_Configs, desc="LLM Loop"):
        model_name = llm_config.model_name
        
        # Construct model output directory path
        model_out_dir = f"{eval_config.output_dir}/{llm_config.company}/{model_name}"

        logger.info(f"Generating judgement file {judgement_file} for LLM {model_name}")

        summaries_jsonl_path = os.path.join(model_out_dir, summary_file)

        if os.path.isfile(summaries_jsonl_path):
            logger.info(f"{summary_file} found for {model_name}")
            summaries_df = pd.read_json(summaries_jsonl_path, lines=True)
            article_summary_df = pd.merge(
                article_df, summaries_df,
                on=BasicSummary.Keys.ARTICLE_ID, how='inner'
            )

            judgements_jsonl_path = os.path.join(model_out_dir, judgement_file)
            open(judgements_jsonl_path, 'w').close()

            calc_and_save_metrics(
                hhem_model, 
                article_summary_df, 
                judgements_jsonl_path,
                eval_name = eval_config.eval_name,
                eval_date = eval_config.eval_date
            )
        else:
            logger.warning(
                f"{summary_file} not found for {model_name}, skipping model"
            )
    logger.info(
        f"Finished generating and saving {judgement_file} for all models"
    )

def calc_and_save_metrics(
        hhem_model: HHEM_2_1_open | HHEM_2_3,
        article_summary_df: pd.DataFrame,
        judgements_jsonl_path: str,
        eval_name: str,
        eval_date: str
    ):
    """
    Produces metrics for Articles and Summaires aligned on article_id

    Output is incrementally saved to the given jsonl file

    Args:
        hhem_model: HHEM model instance (HHEM_2_1_open or HHEM_2_3)
        article_summary_df (DataFrame): contains article and summaries merged 
            on article_id
        judgements_jsonl_path (str): path to JSONL file that contain the HHEM, 
            validity, and  length of each summary
        eval_name (str): name of the evaluation
        eval_date (str): date of the evaluation
    Returns:
        None
    """
    article_texts = article_summary_df[SourceArticle.Keys.TEXT].tolist()
    article_summaries = article_summary_df[BasicSummary.Keys.SUMMARY].tolist()
    summary_uids = article_summary_df[BasicSummary.Keys.SUMMARY_UID].tolist()

    for premise, hypothesis, summary_uid in tqdm(
        zip(article_texts, article_summaries, summary_uids),
        total=len(article_texts),
        desc="HHEM/Judgement Loop"
    ):
        input = (premise, hypothesis)
        hhem_out: HHEMOutput = hhem_model.predict(*input)
        summary_length = len(hypothesis.split())
        is_summary_valid = is_valid_summary(hypothesis)
        metric_record = Judgement(
            eval_name = eval_name,
            eval_date = eval_date,
            summary_uid = summary_uid,
            hhem_version = hhem_model.__str__(),
            hhem_score = hhem_out.score,
            is_valid=is_summary_valid,
            summary_words=summary_length
        )
        append_record_to_jsonl(judgements_jsonl_path, metric_record)

if __name__ == "__main__":
    pass