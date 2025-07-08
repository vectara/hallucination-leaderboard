import hashlib
import json
import os
from datetime import datetime, timezone
from typing import List, Literal, Tuple, Any

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

def prepare_llm(
    eval_config: EvalConfig, # The config for this evaluation run
    llm_config: BasicLLMConfig,  # the particular LLM config in the eval_config
    summary_file: str
) -> Tuple[AbstractLLM, type]: 

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

        # print ("LLM config passed to instantiate_llm: ", json.dumps(llm_config.model_dump(), indent=2))
        
        # update_data = {}
        # for key, value in llm_config.model_dump().items():
        #     if value == None and key in LLM_CONFIG_CLASS.model_fields:
        #         update_data[key] = LLM_CONFIG_CLASS.model_fields[key].default
        #     else: 
        #         update_data[key] = value
        # specific_config = LLM_CONFIG_CLASS(**update_data)
        
        # print ("LLM config after instantiation: ", json.dumps(specific_config.model_dump(), indent=2))
        
        # llm = LLM_CLASS(specific_config) # instantiate the LLM

        # For parameters that are not in per_LLM_configs or are None, use the Non-None values in common_LLM_config
        update_data = {}
        for key, value in eval_config.common_LLM_config.model_dump().items():
            if key not in llm_config.model_dump():
                update_data[key] = value
            elif key in llm_config.model_dump():
                if value is not None and llm_config.model_dump()[key] is None:
                    update_data[key] = value

        # print ("Update data: ", json.dumps(update_data, indent=2))
        if update_data:
            llm_config = llm_config.model_copy(update=update_data)

        llm = LLM_CLASS(llm_config) # instantiate the LLM

        # Create output directory 
        full_output_dir = f"{eval_config.output_dir}/{llm_config.company}/{llm_config.model_name}"
        os.makedirs(full_output_dir, exist_ok=True)

        summaries_jsonl_path = os.path.join(full_output_dir, summary_file)
        llm.summary_file = summaries_jsonl_path

        # Create summary file if it doesn't exist
        if not os.path.isfile(summaries_jsonl_path):
            logger.info(f"Summary file {summary_file} file does not exist, creating...")
            open(summaries_jsonl_path, 'w').close()
        elif os.path.isfile(summaries_jsonl_path) and eval_config.overwrite_summaries:
            logger.info(f"Overwriting previous summaries in summary file {summary_file}")
            open(summaries_jsonl_path, 'w').close()
        else:
            logger.info(f"Appending additional summaries to summary file {summary_file}")
        
        return llm, LLM_SUMMARY_CLASS
    
    except Exception as e:
        logger.error(f"Failed to prepare LLM {llm_config.company}/{llm_config.model_name}: {e}")
        raise

def get_summaries(eval_config: EvalConfig, article_df: pd.DataFrame, summary_file: str):
    """
    Generates summaries for a given model and then save to a jsonl file, overwrite flag will overwrite existing jsonl file
    """
    LLMs_to_be_processed = [llm_config.model_name for llm_config in eval_config.per_LLM_configs]
    logger.info(f"Starting to generate {summary_file} for the following LLMs: {LLMs_to_be_processed}")

    common_LLM_config = eval_config.common_LLM_config

    for llm_config in tqdm(eval_config.per_LLM_configs, desc="LLM Loop"):

        # Instantiate the LLM
        llm, LLM_SUMMARY_CLASS = prepare_llm(eval_config, llm_config, summary_file)

        # print all attributes of llm
        # logger.info(f"LLM attributes: {json.dumps(llm.__dict__, indent=2)}")

        generate_and_save_summaries(
            llm, 
            article_df, 
            eval_config,
            llm_config, 
            LLM_SUMMARY_CLASS
        )
        logger.info(f"Finished generating and saving summaries for LLM {llm_config.model_name} into {llm.summary_file}.")
        
        logger.info("Moving on to next LLM")
    
    logger.info(f"Finished generating and saving summaries for the following LLMs: {LLMs_to_be_processed}")

def generate_and_save_summaries(
        llm: AbstractLLM,
        article_df: pd.DataFrame,
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
        eval_config (EvalConfig): The evaluation configuration
        llm_config (BasicLLMConfig): The LLM configuration
        LLM_SUMMARY_CLASS (type): The class of the summary to be saved. For Pydantic validation.
    Returns:
        None
    """

    current_date = datetime.now(timezone.utc).date().isoformat()
    article_texts = article_df[SourceArticle.Keys.TEXT].tolist()
    article_ids = article_df[SourceArticle.Keys.ARTICLE_ID].tolist()

    logger.info(f"Generating summaries for LLM {llm_config.model_name} and appending to jsonl file {llm.summary_file}")
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

            record_data = {
                'article_id': article_id,
                'summary_uid': summary_uid,
                'summary': summary,
                'eval_name': eval_config.eval_name,
                'eval_date': eval_config.eval_date,
                # **llm_config.model_dump()
                **llm.__dict__
            }

            # do no include prompt in the record
            record_data.pop('prompt', None)
            
            record = LLM_SUMMARY_CLASS(**record_data)
            append_record_to_jsonl(llm.summary_file, record)


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