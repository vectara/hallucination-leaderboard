import hashlib
import json
import os
from datetime import datetime, timezone
from typing import List, Literal, Tuple, Any

import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

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

class TooManyRequestsError(Exception):
    pass

# General purpose check for 429 errors
def is_rate_limit_error(exception):
    # If the exception has a `status_code` attribute
    if hasattr(exception, "status_code") and exception.status_code == 429:
        return True
    # If the exception has a string message containing 429 or rate limit
    if "429" in str(exception) or "rate limit" in str(exception).lower():
        return True
    return False

def prepare_llm(
    eval_config: EvalConfig,
    llm_config: BasicLLMConfig,
) -> Tuple[AbstractLLM, type, str]: 

    try:
        llm_registry = MODEL_REGISTRY.get(llm_config.company)
        LLM_CLASS = llm_registry["LLM_class"]
        LLM_SUMMARY_CLASS = llm_registry["summary_class"]
        
        if LLM_CLASS is None:
            raise Exception(ModelInstantiationError.NOT_REGISTERED.format(
                model_name=llm_config.model_name,
                company=llm_config.company,
            ))
        
        for common_key in eval_config.common_LLM_config.model_fields_set:
            if common_key not in llm_config.model_fields_set:
                llm_config = llm_config.model_copy(update={common_key: getattr(eval_config.common_LLM_config, common_key)})

        llm = LLM_CLASS(llm_config)

        full_output_dir = ""
        if llm_config.date_code == "" or llm_config.date_code == None:
            full_output_dir = f"{eval_config.output_dir}/{llm_config.company}/{llm_config.model_name}"
        else:
            full_output_dir = f"{eval_config.output_dir}/{llm_config.company}/{llm_config.model_name}-{llm_config.date_code}"

        os.makedirs(full_output_dir, exist_ok=True)

        summaries_jsonl_path = os.path.join(full_output_dir, eval_config.summary_file)
        if not os.path.isfile(summaries_jsonl_path):
            logger.info(f"Summary file {summaries_jsonl_path} does not exist, creating...")
            open(summaries_jsonl_path, 'w').close()
        elif os.path.isfile(summaries_jsonl_path) and eval_config.overwrite_summaries:
            if not input(f"Are you sure you want to overwrite previous summaries in {summaries_jsonl_path}? (upper case YES to continue)").upper() == "YES":
                raise Exception("User chose not to overwrite previous summaries. Abort to avoid data loss.")
            else: 
                logger.info(f"Overwriting previous summaries in summary file {summaries_jsonl_path}")
                llm.prepare_for_overwrite(summaries_jsonl_path, eval_config.eval_date)
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

        llm, LLM_SUMMARY_CLASS, summaries_jsonl_path = prepare_llm(eval_config, llm_config)

        llm_alias = ""
        if llm_config.date_code == "" or llm_config.date_code == None:
            llm_alias = f"{llm_config.model_name}"
        else:
            llm_alias = f"{llm_config.model_name}-{llm_config.date_code}"

        logger.info(f"Generating summaries for LLM {llm_alias} and saving to jsonl file {summaries_jsonl_path}")
        
        if llm_config.threads == 1:
            generate_summaries_for_one_llm(
                llm, 
                article_df, 
                eval_config.eval_name,
                eval_config.eval_date,
                summaries_jsonl_path,
                LLM_SUMMARY_CLASS
            )
        elif llm_config.threads > 1:
            def llm_factory(eval_config, llm_config):
                try:
                    llm_registry = MODEL_REGISTRY.get(llm_config.company)
                    LLM_CLASS = llm_registry["LLM_class"]
                    
                    if LLM_CLASS is None:
                        raise Exception(ModelInstantiationError.NOT_REGISTERED.format(
                            model_name=llm_config.model_name,
                            company=llm_config.company,
                        ))

                    for common_key in eval_config.common_LLM_config.model_fields_set:
                        if common_key not in llm_config.model_fields_set:
                            llm_config = llm_config.model_copy(update={common_key: getattr(eval_config.common_LLM_config, common_key)})
                    llm = LLM_CLASS(llm_config)
                    return llm
                
                except Exception as e:
                    logger.error(f"Failed to prepare LLM {llm_config.company}/{llm_config.model_name}: {e}")
                    raise

            generate_summaries_for_one_llm_multithreaded(
                llm_factory=llm_factory,
                article_df=article_df,
                eval_config=eval_config,
                llm_config=llm_config,
                summaries_jsonl_path=summaries_jsonl_path,
                LLM_SUMMARY_CLASS=LLM_SUMMARY_CLASS,
                max_workers=llm_config.threads
            )
        else:
            raise Exception("Improper Thread Number. Aborting.")
        logger.info(f"Finished generating and saving summaries for LLM {llm_alias} into {summary_file}.")
        
        logger.info("Moving on to next LLM")
    
    logger.info(f"Finished generating and saving summaries for the following LLMs: {LLMs_to_be_processed}")

# TODO: Docs
def generate_summaries_for_one_llm_multithreaded(
        llm_factory,
        article_df,
        eval_config,
        llm_config,
        summaries_jsonl_path,
        LLM_SUMMARY_CLASS,
        max_workers: int
    ):
    
    current_date = datetime.now(timezone.utc).date().isoformat()
    article_texts = article_df[SourceArticle.Keys.TEXT].tolist()
    article_ids   = article_df[SourceArticle.Keys.ARTICLE_ID].tolist()
    eval_name=eval_config.eval_name
    eval_date=eval_config.eval_date

    # WRITER THREAD
    q = Queue()
    writer_done = threading.Event()

    def writer():
        with open(summaries_jsonl_path, "a", encoding="utf8") as f:
            while not (writer_done.is_set() and q.empty()):
                try:
                    record = q.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    f.write(json.dumps(record) + "\n")
                except Exception as e:
                    logger.error(f"Failed to write record: {e}")
                q.task_done()

    wt = threading.Thread(target=writer)
    wt.start()

    
    def worker(article_text, article_id):
        try:
            llm = llm_factory(eval_config, llm_config)
            with llm as m:
                @retry(
                    retry=retry_if_exception_type(is_rate_limit_error),
                    wait=wait_exponential(multiplier=1, min=2, max=60),
                    stop=stop_after_attempt(10)
                )
                def summarize_with_retry(text):
                    return m.try_to_summarize_one_article(text)
    
                summary = summarize_with_retry(article_text)
    
                summary_uid = generate_summary_uid(
                    m.model_fullname,
                    summary,
                    current_date
                )
    
                record_data = {
                    "article_id": article_id,
                    "summary_uid": summary_uid,
                    "summary": summary,
                    "eval_name": eval_name,
                    "summary_date": eval_date,
                    **m.__dict__
                }
                record_data.pop("prompt", None)
    
                record = LLM_SUMMARY_CLASS(**record_data)
                q.put(record.model_dump())
    
        except Exception as e:
            summary_uid = generate_summary_uid(
                m.model_fullname,
                "THREAD ERROR",
                current_date
            )
            error_record = {
                "article_id": article_id,
                "summary_uid": summary_uid,
                "summary": f"THREAD ERROR",
                "eval_name": eval_name,
                "summary_date": eval_date,
                **m.__dict__
            }
            error_record.pop("prompt", None)
            error_record = LLM_SUMMARY_CLASS(**error_record)
            q.put(error_record)
            logger.error(f"Worker failed for article_id={article_id}: {e}")

    # THREAD EXECUTOR
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for text, aid in zip(article_texts, article_ids):
            futures.append(ex.submit(worker, text, aid))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Summaries"):
            pass

    writer_done.set()
    wt.join()

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