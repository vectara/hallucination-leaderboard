import os
from typing import List, Literal, Tuple, Any

import pandas as pd
from tqdm import tqdm

from datetime import datetime, timezone
import hashlib

from .. data_model import SourceArticle, ModelInstantiationError, BasicSummary
from .. json_utils import file_exists, append_record_to_jsonl
from .. LLMs import AbstractLLM, MODEL_REGISTRY, BasicLLMConfig
from .. Logger import logger


"""
Loops through all given LLMs and requests a summary for the give article
dataframe. 

Global Variables:
    SUMMARY_FILE

Functions:
    run(models)
    generate_and_save_summaries(model, article_df, json_path)
    generate_summary_uid(model_name, date_code, summary_text, date)
"""

SUMMARY_FILE = "summaries.jsonl"

def instantiate_llm(llm_config: BasicLLMConfig) -> Tuple[AbstractLLM, type, type]: # TODO: how to type the classes? 
    try:
        llm_registry = MODEL_REGISTRY.get(llm_config.model_name)
        LLM_CLASS = llm_registry["LLM_class"]
        if LLM_CLASS is None:
            raise ModelInstantiationError.NOT_REGISTERED.format(
                model_name=llm_config.model_name,
                company=llm_config.company,
            )
        llm = LLM_CLASS(llm_config) # instantiate the LLM
        return llm, llm_registry["config_class"], llm_registry["summary_class"]
    except Exception as e:
        logger.error(f"Failed to instantiate model {llm_config.company}/{llm_config.params.model_name}: {e}")
        raise

def run(llm_configs: List[BasicLLMConfig], article_df: pd.DataFrame, ow=False):
    """
    Generates summaries for a given model if the corresponding jsonl file does 
    not exist, overwrite flag will overwrite existing jsonl file

    Args:
        llm_configs (list[LLMConfig]): list of LLM configurations
        article_df (pd.DataFrame): articles to be summarized
        ow (bool): flag to specify if JSON should still be created if exists

    Returns:
        None
    """

    logger.info(f"Starting to generate {SUMMARY_FILE}")
    if ow:
        logger.info(
            f"Overwrite flag enabled"
        )

    for llm_config in tqdm(llm_configs, desc="LLM Loop"):

        llm, LLM_CONFIG_CLASS, LLM_SUMMARY_CLASS = instantiate_llm(llm_config)
        llm_name = llm_config.model_name
        llm_out_dir = llm.model_output_dir

        logger.info(f"Generating {SUMMARY_FILE} for {llm_name}")

        jsonl_file = f"{SUMMARY_FILE}"
        summaries_jsonl_path = os.path.join(llm_out_dir, jsonl_file)

        if not file_exists(summaries_jsonl_path):
            logger.info(f"{SUMMARY_FILE} file does not exist, creating...")
            open(summaries_jsonl_path, 'w').close()
        elif file_exists(summaries_jsonl_path) and ow:
            logger.info(f"Overwriting previous data in {SUMMARY_FILE}")
            open(summaries_jsonl_path, 'w').close()
        else:
            logger.info(f"Adding additional data to {SUMMARY_FILE}")

        generate_and_save_summaries(llm, article_df, summaries_jsonl_path, llm_config, LLM_SUMMARY_CLASS)
        logger.info(f"Finished generating and saving summaries for {llm_name} into {summaries_jsonl_path}.")
        
        logger.info("Moving on to next LLM")
    
    logger.info(f"Finished generating and saving summaries for all LLMs")

def generate_and_save_summaries(
        llm: AbstractLLM,
        article_df: pd.DataFrame,
        jsonl_path: str,
        llm_config: BasicLLMConfig,  # This will be the specific config class instance
        LLM_SUMMARY_CLASS: type
    ):
    """
    Produces summaries for all articles and saves them to the given jsonl file.
    Saving is performed incrementally.

    Args:
        llm (AbstractLLM): An instance of an LLM following the AbstractLLM interface
        article_df (pd.DataFrame): Article data
        jsonl_path (str): path for the new jsonl file

    Returns:
        None
    """

    current_date = datetime.now(timezone.utc).date().isoformat()
    article_texts = article_df[SourceArticle.Keys.TEXT].tolist()
    article_ids = article_df[SourceArticle.Keys.ARTICLE_ID].tolist()

    # Get the base fields that are common to all configs
    common_config_fields = {
        'timestamp': current_date,
        'llm': llm.model_name,
        'temperature': llm.temperature,
        'max_tokens': llm.max_tokens,
    }

    if llm.thinking_tokens is not None:
        common_config_fields['thinking_tokens'] = llm.thinking_tokens
    if llm.execution_mode is not None:
        common_config_fields['execution_mode'] = llm.execution_mode
    
    # Extract additional fields from the specific config class that are not in BasicLLMConfig
    # Get all fields from the config
    config_dict = llm_config.model_dump()
    
    # Get fields from BasicLLMConfig to exclude them
    basic_config_fields = set(BasicLLMConfig.model_fields.keys())
    
    # Extract additional fields that are specific to this config class
    llm_specific_config_fields = {
        key: value for key, value in config_dict.items() 
        if key not in basic_config_fields
    }
    logger.info("Appending to jsonl file")
    with llm as m: 
        for article_text, article_id in tqdm(
            zip(article_texts, article_ids),
            total=len(article_texts),
            desc="Article Loop"
        ):
            summary = m.summarize_clean_wait(article_text)
            summary_uid = generate_summary_uid(
                llm.model_fullname,
                summary,
                current_date
            )
            
            # Combine base fields with article-specific data and additional config fields
            record_data = {
                **common_config_fields,
                'summary_uid': summary_uid,
                'summary': summary,
                'article_id': article_id,
                **llm_specific_config_fields  # Add any additional fields from the specific llm
            }
            
            record = LLM_SUMMARY_CLASS(**record_data)
            append_record_to_jsonl(jsonl_path, record)


def generate_summary_uid(
        model_name: str, summary_text: str, date: str
    ) -> str:
    """
    Generates a hash for the summary using the model name, date code, summary 
    text, date and time.

    Args:
        model_name (str): name of model
        date_code (str): date code of model
        summary_text (str): summary generated by model
        date (str): current date

    Returns
        str: hash of summary generation
    """
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    combined_string = (
        f"{model_name}|{summary_text.strip()}|{date}|{current_time}"
    )
    return hashlib.md5(combined_string.encode('utf-8')).hexdigest()
            
if __name__ == "__main__":
    pass