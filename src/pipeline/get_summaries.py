import os
from typing import List, Literal, Tuple, Any
import json
from datetime import datetime, timezone
import hashlib

import pandas as pd
from tqdm import tqdm

from .. data_model import SourceArticle, ModelInstantiationError, BasicLLMConfig, BasicSummary, EvalConfig
from .. json_utils import append_record_to_jsonl
from .. LLMs import AbstractLLM, MODEL_REGISTRY
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
        llm_registry = MODEL_REGISTRY.get(llm_config.company)
        LLM_CLASS = llm_registry["LLM_class"]
        LLM_CONFIG_CLASS = llm_registry["config_class"]
        LLM_SUMMARY_CLASS = llm_registry["summary_class"]
        
        if LLM_CLASS is None:
            raise Exception(ModelInstantiationError.NOT_REGISTERED.format(
                model_name=llm_config.model_name,
                company=llm_config.company,
            ))
        
        update_data = {}
        for key, value in llm_config.model_dump().items():
            if value == None and key in LLM_CONFIG_CLASS.model_fields:
                update_data[key] = LLM_CONFIG_CLASS.model_fields[key].default
            else: 
                update_data[key] = value
        specific_config = LLM_CONFIG_CLASS(**update_data)
        
        llm = LLM_CLASS(specific_config) # instantiate the LLM
        return llm, LLM_CONFIG_CLASS, LLM_SUMMARY_CLASS
    except Exception as e:
        logger.error(f"Failed to instantiate model {llm_config.company}/{llm_config.model_name}: {e}")
        raise

def run(eval_config: EvalConfig, article_df: pd.DataFrame):
    """
    Generates summaries for a given model and then save to a jsonl file, overwrite flag will overwrite existing jsonl file
    """
    logger.info(f"Starting to generate {SUMMARY_FILE}")
    # if eval_config.overwrite_summaries:
    #     logger.info(
    #         f"Overwrite flag enabled"
    #     )

    for llm_config in tqdm(eval_config.LLM_Configs, desc="LLM Loop"):        
        # For parameters that are not in llm_config, use the values in eval_config
        update_data = {}
        
        for key, value in eval_config.model_dump().items():
            if key not in llm_config.model_dump():
                update_data[key] = value
                # logger.info(f"For parameter {key}, using LLM-agnostic value {value}")
            elif key in llm_config.model_dump():
                if value is not None and llm_config.model_dump()[key] is None:
                    update_data[key] = value
                    # logger.info(f"For parameter {key}, using LLM-agnostic value {value}")
            #     else:
            #         logger.info(f"For parameter {key}, using LLM-specific value {value}")
            # else:
            #     logger.info(f"For parameter {key}, using LLM-specific value {value}")
        
        if update_data:
            llm_config = llm_config.model_copy(update=update_data)

        # logger.info(f"LLM config after merging: {json.dumps(llm_config.model_dump(exclude_none=True), indent=2)}")

        # Instantiate the LLM
        llm, LLM_CONFIG_CLASS, LLM_SUMMARY_CLASS = instantiate_llm(llm_config)

        # print all attributes of llm
        # logger.info(f"LLM attributes: {json.dumps(llm.__dict__, indent=2)}")

        llm_name = llm_config.model_name
        llm_out_dir = llm.model_output_dir

        logger.info(f"Generating {SUMMARY_FILE} for {llm_name}")

        jsonl_file = f"{SUMMARY_FILE}"
        summaries_jsonl_path = os.path.join(llm_out_dir, jsonl_file)

        if not os.path.isfile(summaries_jsonl_path):
            logger.info(f"{SUMMARY_FILE} file does not exist, creating...")
            open(summaries_jsonl_path, 'w').close()
        elif os.path.isfile(summaries_jsonl_path) and eval_config.overwrite_summaries:
            logger.info(f"Overwriting previous data in {SUMMARY_FILE}")
            open(summaries_jsonl_path, 'w').close()
        else:
            logger.info(f"Adding additional data to {SUMMARY_FILE}")

        generate_and_save_summaries(
            llm, 
            article_df, 
            summaries_jsonl_path, 
            eval_config,
            llm_config, 
            LLM_SUMMARY_CLASS
        )
        logger.info(f"Finished generating and saving summaries for {llm_name} into {summaries_jsonl_path}.")
        
        logger.info("Moving on to next LLM")
    
    logger.info(f"Finished generating and saving summaries for all LLMs")

def generate_and_save_summaries(
        llm: AbstractLLM,
        article_df: pd.DataFrame,
        jsonl_path: str,
        eval_config: EvalConfig,
        llm_config: BasicLLMConfig,  # TODO: How can I declear the type as of BasicLLMConfig or any child of BasicLLMConfig?
        LLM_SUMMARY_CLASS: type
    ):
    """
    Produces summaries for all articles and saves them to the given jsonl file.
    Saving is performed incrementally.

    Args:
        llm (AbstractLLM): An instance of an LLM following the AbstractLLM interface
        article_df (pd.DataFrame): Article data where each row is an instance of SourceArticle
        jsonl_path (str): path for the new jsonl file
        eval_config (EvalConfig): The evaluation configuration
        llm_config (BasicLLMConfig): The LLM configuration
        LLM_SUMMARY_CLASS (type): The class of the summary to be saved. For Pydantic validation.
    Returns:
        None
    """

    current_date = datetime.now(timezone.utc).date().isoformat()
    article_texts = article_df[SourceArticle.Keys.TEXT].tolist()
    article_ids = article_df[SourceArticle.Keys.ARTICLE_ID].tolist()

    # # Get the base fields that are common to all configs
    # common_config_fields = {
    #     'timestamp': current_date,
    #     'llm': llm.model_name,
    #     'temperature': llm.temperature,
    #     'max_tokens': llm.max_tokens,
    # }

    # # Extract additional fields from the specific config class that are not in BasicLLMConfig
    # Get all fields from the config
    # config_dict = llm_config.model_dump()
    
    # # Get fields from BasicLLMConfig to exclude them
    # basic_config_fields = set(BasicLLMConfig.model_fields.keys())

    # # remove prompt, output_dir, and min_throttle_time from basic_config_fields
    # basic_config_fields.discard('prompt')
    # basic_config_fields.discard('output_dir')
    # basic_config_fields.discard('min_throttle_time')

    # # Remove any basic_config_fields that are None 
    # basic_config_fields = {
    #     key: value for key, value in basic_config_fields.items()
    #     if value is not None
    # }
    
    # Extract additional fields that are specific to this config class
    # llm_specific_config_fields = {
    #     key: value for key, value in config_dict.items() 
    #     if key not in basic_config_fields
    # }
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
            # record_data = {
            #     **common_config_fields,
            #     'summary_uid': summary_uid,
            #     'summary': summary,
            #     'article_id': article_id,
            #     **llm_specific_config_fields  # Add any additional fields from the specific llm
            # }

            record_data = {
                'article_id': article_id,
                'summary_uid': summary_uid,
                'summary': summary,
                'eval_name': eval_config.eval_name,
                **llm_config.model_dump()
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