import os
from datetime import datetime, timezone
from typing import Literal
import pandas as pd
from tqdm import tqdm

from .. analytics import is_valid_summary
from .. data_model import BasicJudgment, SourceArticle, EvalConfig, BasicSummary
from .. HHEM_2_x import HHEM_2_1_open, HHEM_2_3, HHEMOutput
from .. HDM_2 import HDM2
from .. json_utils import append_record_to_jsonl
from .. Logger import logger

# TODO: Update to make it model name agnostic?
def get_hhem_model(hhem_version: Literal["2.1-open", "2.3", "HDM-2"]):
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
    elif hhem_version == "HDM-2":
        return HDM2()
    else:
        raise ValueError(f"Unsupported HHEM version: {hhem_version}. Supported versions: 2.1-open, 2.3")

def get_judgments(eval_config: EvalConfig, article_df: pd.DataFrame):
    """
    Generate judgment scores for summaries produced by LLMs.

    Args:
        eval_config (EvalConfig): evaluation configuration
        article_df (pd.DataFrame): dataframe containing articles
        summary_file (str): name of the summary file
        judgment_file (str): name of the judgment file

    Returns:
        None
    """

    summary_file = eval_config.summary_file
    judgment_file = eval_config.judgment_file


    LLMs_to_be_processed = [llm_config.model_name for llm_config in eval_config.per_LLM_configs]
    logger.info(f"Starting to generate {judgment_file} scores for the following LLMs: {LLMs_to_be_processed}")

    for llm_config in tqdm(eval_config.per_LLM_configs, desc="LLM Loop"):
        model_name = llm_config.model_name

        llm_alias = ""
        if llm_config.date_code == "" or llm_config.date_code == None:
            llm_alias = f"{llm_config.model_name}"
        else:
            llm_alias = f"{llm_config.model_name}-{llm_config.date_code}"
        
        # Construct model output directory path
        model_out_dir = ""
        if llm_config.date_code == "" or llm_config.date_code == None:
            model_out_dir = os.path.join(
                eval_config.output_dir, 
                llm_config.company, 
                model_name
            )
        else:
            model_out_dir = os.path.join(
                eval_config.output_dir, 
                llm_config.company, 
                f"{llm_config.model_name}-{llm_config.date_code}"
            )
        summaries_jsonl_path = os.path.join(model_out_dir, summary_file)

        logger.info(f"Generating judgment file {judgment_file} for LLM {llm_alias}")

        if os.path.isfile(summaries_jsonl_path):
            summaries_df = pd.read_json(summaries_jsonl_path, lines=True)
            
            # Create judgment file
            judgments_jsonl_path = os.path.join(model_out_dir, judgment_file)
            open(judgments_jsonl_path, 'w').close()
            
            # Generate judgments
            hhem_model = get_hhem_model(eval_config.hhem_version)
            article_summary_df = pd.merge(
                article_df, summaries_df,
                on=BasicSummary.Keys.ARTICLE_ID, how='inner'
            )
            generate_judgments(
                hhem_model, article_summary_df, judgments_jsonl_path, eval_config.eval_name, eval_config.eval_date
            )
            logger.info(f"Finished judging summaries produced by LLM {llm_alias} and saved to {judgment_file}")
        else:
            logger.warning(
                f"Summary file {summaries_jsonl_path} not found for LLM {llm_alias}, skipping LLM"
            )
    logger.info(f"Finished generating {judgment_file} scores for the following LLMs: {LLMs_to_be_processed}")

def generate_judgments(
    hhem_model: HHEM_2_1_open | HHEM_2_3,
    article_summary_df: pd.DataFrame,
    judgments_jsonl_path: str,
    eval_name: str,
    eval_date: str,
):
    """
    Generate judgment scores for summaries using HHEM model.

    Args:
        hhem_model (HHEMOutput): HHEM model for generating judgments
        article_summary_df (pd.DataFrame): dataframe containing articles and summaries
        judgments_jsonl_path (str): path to save judgment results
        eval_name (str): name of the evaluation
        eval_date (str): date of the evaluation

    Returns:
        None
    """
    for _, row in tqdm(article_summary_df.iterrows(), total=len(article_summary_df), 
                       desc="HHEM/Judgment Loop"):
        hypothesis = row[BasicSummary.Keys.SUMMARY]
        word_count = len(hypothesis.split())
        valid_summary = is_valid_summary(hypothesis)
        metric_record = BasicJudgment(
            eval_name=eval_name,
            judgment_date=eval_date,
            summary_uid=row[BasicSummary.Keys.SUMMARY_UID],
            hhem_version=hhem_model.__str__(),
            hhem_score=hhem_model.predict(row[SourceArticle.Keys.TEXT], hypothesis).score,
            is_valid=valid_summary,
            word_count = word_count
        )
        append_record_to_jsonl(judgments_jsonl_path, metric_record)

if __name__ == "__main__":
    pass