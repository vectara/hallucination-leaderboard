"""Summarization pipeline for generating LLM summaries.

This module provides functionality for generating summaries from source articles
using configured LLM models. Supports both single-threaded and multi-threaded
execution modes with automatic retry logic for rate-limited API calls.

The pipeline iterates through configured LLM models, generates summaries for
each article in the dataset, and saves results incrementally to JSONL files.

Classes:
    TooManyRequestsError: Custom exception for rate limit errors.

Functions:
    is_rate_limit_error: Check if an exception is a rate limit error.
    prepare_llm: Initialize an LLM instance with configuration.
    get_summaries: Main entry point for generating summaries.
    generate_summaries_for_one_llm: Single-threaded summary generation.
    generate_summaries_for_one_llm_multithreaded: Multi-threaded summary generation.
    generate_summary_uid: Generate unique identifier for a summary.
"""

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

class TooManyRequestsError(Exception):
    """Custom exception for HTTP 429 Too Many Requests errors.

    Raised when an API rate limit is exceeded. Used as a signal for
    retry logic to wait and attempt the request again.
    """

    pass


def is_rate_limit_error(exception):
    """Check if an exception indicates a rate limit error.

    Performs a general-purpose check for HTTP 429 errors by examining
    the exception's status_code attribute and string representation.

    Args:
        exception: The exception to check for rate limit indicators.

    Returns:
        True if the exception appears to be a rate limit error, False otherwise.
    """
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
    """Prepare an LLM instance for summary generation.

    Initializes an LLM instance from the model registry, applies common
    configuration settings, creates the output directory structure, and
    handles summary file creation or overwrite logic.

    Args:
        eval_config: Evaluation configuration containing output paths and
            common LLM settings.
        llm_config: Model-specific configuration for the LLM to prepare.

    Returns:
        A tuple containing:
            - llm: The initialized AbstractLLM instance.
            - LLM_SUMMARY_CLASS: The Pydantic model class for summary records.
            - summaries_jsonl_path: Path to the output JSONL file.

    Raises:
        Exception: If the model is not registered or user declines overwrite.
    """
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
    """Generate summaries for all configured LLMs and save to JSONL files.

    Main entry point for the summarization pipeline. Iterates through all
    LLM configurations in eval_config, generates summaries for each article
    in the dataset, and saves results incrementally. Supports both single-
    threaded and multi-threaded execution based on the threads setting.

    Args:
        eval_config: Evaluation configuration containing LLM configs, output
            paths, and execution settings.
        article_df: DataFrame containing source articles with columns for
            article text and article IDs.

    Raises:
        Exception: If an invalid thread count is specified.
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

def generate_summaries_for_one_llm_multithreaded(
        llm_factory,
        article_df,
        eval_config,
        llm_config,
        summaries_jsonl_path,
        LLM_SUMMARY_CLASS,
        max_workers: int
    ):
    """Generate summaries using multiple worker threads.

    Parallelizes summary generation across multiple threads for improved
    throughput. Uses a dedicated writer thread for thread-safe file I/O
    and implements retry logic with exponential backoff for rate-limited
    API calls.

    Args:
        llm_factory: Callable that creates new LLM instances for each worker.
            Signature: (eval_config, llm_config) -> AbstractLLM.
        article_df: DataFrame containing source articles with text and IDs.
        eval_config: Evaluation configuration with eval_name and eval_date.
        llm_config: Model-specific configuration for the LLM.
        summaries_jsonl_path: Path to the output JSONL file.
        LLM_SUMMARY_CLASS: Pydantic model class for summary records.
        max_workers: Maximum number of concurrent worker threads.

    Note:
        Each worker creates its own LLM instance to avoid thread-safety
        issues with shared client connections.
    """
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
    """Generate summaries for all articles using a single LLM instance.

    Iterates through all articles in the DataFrame, generates a summary for
    each using the provided LLM, and saves results incrementally to a JSONL
    file. Uses a context manager to handle LLM setup and teardown.

    Args:
        llm: The initialized LLM instance to use for generation.
        article_df: DataFrame containing source articles with text and IDs.
        eval_name: Name identifier for this evaluation run.
        eval_date: Date identifier for this evaluation run.
        summaries_jsonl_path: Path to the output JSONL file.
        LLM_SUMMARY_CLASS: Pydantic model class for summary records.
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
    """Generate a unique identifier hash for a summary.

    Creates an MD5 hash from the combination of model name, summary text,
    date, and current time to ensure uniqueness even for identical summaries
    generated at different times.

    Args:
        model_name: Full name of the model that generated the summary.
        summary_text: The generated summary text content.
        date: Date string for the evaluation run.

    Returns:
        A 32-character hexadecimal MD5 hash string.
    """
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    combined_string = (
        f"{model_name}|{summary_text.strip()}|{date}|{current_time}"
    )
    return hashlib.md5(combined_string.encode('utf-8')).hexdigest()
            
if __name__ == "__main__":
    pass