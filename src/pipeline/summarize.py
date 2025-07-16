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
    llm_config: BasicLLMConfig,  # an LLM-specific config in the eval_config. It should be of any class inherited from BasicLLMConfig
) -> Tuple[AbstractLLM, type, str]: 

    try:
        llm_registry = MODEL_REGISTRY.get(llm_config.company)
        LLM_CLASS = llm_registry["LLM_class"]
        # LLM_CONFIG_CLASS = llm_registry["config_class"]
        LLM_SUMMARY_CLASS = llm_registry["summary_class"]
        
        if LLM_CLASS is None:
            raise Exception(ModelInstantiationError.NOT_REGISTERED.format(
                model_name=llm_config.model_name,
                company=llm_config.company,
            ))
        
        # print ("eval_config.common_LLM_config: ", eval_config.common_LLM_config.model_dump_json(indent=2))
        # print ("llm_config: ", llm_config.model_dump_json(indent=2))

        # Replace fields that are not set in per-llm config with those set (not default) in common-llm config
        for common_key in eval_config.common_LLM_config.model_fields_set:
            if common_key not in llm_config.model_fields_set:
                llm_config = llm_config.model_copy(update={common_key: getattr(eval_config.common_LLM_config, common_key)})

        # print ("llm_config after brushing using per_LLM_config: ", llm_config.model_dump_json(indent=2))

        llm = LLM_CLASS(llm_config) # instantiate the LLM

        # Create output directory 
        full_output_dir = f"{eval_config.output_dir}/{llm_config.company}/{llm_config.model_name}"
        os.makedirs(full_output_dir, exist_ok=True)

        summaries_jsonl_path = os.path.join(full_output_dir, eval_config.summary_file)
        if not os.path.isfile(summaries_jsonl_path):
            logger.info(f"Summary file {summaries_jsonl_path} does not exist, creating...")
            open(summaries_jsonl_path, 'w').close()
        elif os.path.isfile(summaries_jsonl_path) and eval_config.overwrite_summaries:
            # warning that we do not recommend overwriting summaries. Type YES is your wanna continue 
            if not input(f"Are you sure you want to overwrite previous summaries in {summaries_jsonl_path}? (upper case YES to continue)").upper() == "YES":
                raise Exception("User chose not to overwrite previous summaries. Abort to avoid data loss.")
            else: 
                logger.info(f"Overwriting previous summaries in summary file {summaries_jsonl_path}")
                llm.prepare_for_overwrite(summaries_jsonl_path)
        else:
            logger.info(f"Appending additional summaries to summary file {summaries_jsonl_path}")
        
        return llm, LLM_SUMMARY_CLASS, summaries_jsonl_path
    
    except Exception as e:
        logger.error(f"Failed to prepare LLM {llm_config.company}/{llm_config.model_name}: {e}")
        raise

def get_summaries(
        eval_config: EvalConfig, 
        article_df: pd.DataFrame):
    """
    Generates summaries for a given model and then save to a jsonl file, overwrite flag will overwrite existing jsonl file
    """

    summary_file = eval_config.summary_file

    LLMs_to_be_processed = [llm_config.model_name for llm_config in eval_config.per_LLM_configs]

    for llm_config in tqdm(eval_config.per_LLM_configs, desc="LLM Loop"):

        # Instantiate the LLM
        llm, LLM_SUMMARY_CLASS, summaries_jsonl_path = prepare_llm(eval_config, llm_config)

        # print all attributes of llm
        # logger.info(f"LLM attributes: {json.dumps(llm.__dict__, indent=2)}")

        logger.info(f"Generating summaries for LLM {llm_config.model_name} and saving to jsonl file {summaries_jsonl_path}")

        generate_summaries_for_one_llm(
            llm, 
            article_df, 
            eval_config.eval_name,
            eval_config.eval_date,
            summaries_jsonl_path,
            LLM_SUMMARY_CLASS
        )
        logger.info(f"Finished generating and saving summaries for LLM {llm_config.model_name} into {summary_file}.")
        
        logger.info("Moving on to next LLM")
    
    logger.info(f"Finished generating and saving summaries for the following LLMs: {LLMs_to_be_processed}")

def generate_summaries_for_one_llm(
        llm: AbstractLLM,
        article_df: pd.DataFrame,
        eval_name: str, 
        eval_date: str, 
        summaries_jsonl_path, 
        LLM_SUMMARY_CLASS: type
    ):
    """
    Produces summaries for all articles and saves them to the given jsonl file.
    Saving is performed incrementally.
    """
    current_date = datetime.now(timezone.utc).date().isoformat()
    article_texts = article_df[SourceArticle.Keys.TEXT].tolist()
    article_ids = article_df[SourceArticle.Keys.ARTICLE_ID].tolist()

    with llm as m: 
        # FIXME: It's better to iterate on rows of a sliced Pandas.DF['col1', 'col2']
        for article_text, article_id in tqdm(
            zip(article_texts, article_ids),
            total=len(article_texts),
            desc="Article Loop"
        ):
            summary = m.try_to_summarize_one_article(article_text)
            summary_uid = generate_summary_uid(
                llm.model_fullname,
                summary,
                current_date
            )

            record_data = {
                'article_id': article_id,
                'summary_uid': summary_uid,
                'summary': summary,
                'eval_name': eval_name,
                'summary_date': eval_date,
                # **llm_config.model_dump()
                **llm.__dict__ # FIXME: should we use something model_dump here? pydantic.basemodel.__dict__ is the old way
            }

            # do no include prompt in the record
            record_data.pop('prompt', None)
            
            record = LLM_SUMMARY_CLASS(**record_data)
            append_record_to_jsonl(summaries_jsonl_path, record)


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