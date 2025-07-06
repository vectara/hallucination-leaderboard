from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum
from datetime import datetime
class SourceArticle(BaseModel):
    """
    Representation of an Article record from LB dataset

    Fields
        article_id (int): unique id of article
        text (str): content text of article
        dataset (str): dataset that article belongs to (TODO: VERIFY TRUE)
    """
    article_id: int
    text: str
    dataset: str

    class Keys:
        ARTICLE_ID = "article_id"
        TEXT = "text"
        DATASET = "dataset"

class BasicJudgment(BaseModel):
    """
    Representation of Judgments/Metrics for the Summary of an Article. An LLM may have its own judgment class that inherits from this class.

    Fields:
        eval_name (str): name of the evaluation
        eval_date (str): date of the evaluation
        summary_uid (str): hash for this summary
        hhem_version (str): version of hhem applied for hhem score
        hhem_score (float): Hughes Hallucination Evaluation Metric (HHEM)
        valid (bool): Validity of the summary, defined in is_valid_summary
        summary_words (int): word count of summary

    Note: 
        Forrest chose to disable datecode in Judgment class because datacode is a property of the LLM, not the summary.
    """
    eval_name: str
    eval_date: str
    # date_code: str | None = None # FIXME: No need for date_code in judgment class because aggregate.py loads datecode from summaries.jsonl
    summary_uid: str
    hhem_version: str
    hhem_score: float
    is_valid: bool
    summary_words: int

    class Keys:
        EVAL_NAME = "eval_name"
        EVAL_DATE = "eval_date"
        # DATE_CODE = "date_code"
        SUMMARY_UID = "summary_uid"
        HHEM_VERSION = "hhem_version"
        HHEM_SCORE = "hhem_score"
        IS_VALID = "is_valid"
        SUMMARY_WORDS = "summary_words"

class Stats(BaseModel):
    """
    Per-LLM stats aggregated from Judgments/Metrics for summaries produced by the LLM on a given dataset and under a given evaluation configuration.

    Fields:
        eval_name (str): name of the evaluation
        eval_date (str): date the stats were performed
        model_name (str): name of the LLM
        date_code (str): date code of the LLM, if applicable
        hhem_version (str): version of HHEM used for this evaluation
        hallucination_rate (float): hallucination rate on all summaries
        confidence_interval (float): variation in the hallucination rate
        answer_rate (float): For all summaries what percentage of them were
            valid
        avg_summary_words (float): Average number of words in all valid
            summaries
    """
    eval_name: str
    eval_date: str
    model_name: str
    date_code: str | None = None
    hhem_version: str
    hallucination_rate: float
    confidence_interval: float
    answer_rate: float
    avg_summary_words: float

    class Keys:
        EVAL_NAME = "eval_name"
        EVAL_DATE = "eval_date"
        MODEL_NAME = "model_name"
        DATE_CODE = "date_code"
        HHEM_VERSION = "hhem_version"
        HALLUCINATION_RATE = "hallucination_rate"
        CONFIDENCE_INTERVAL = "confidence_interval"
        ANSWER_RATE = "answer_rate"
        AVG_SUMMARY_WORDS = "avg_summary_words"


default_prompt = """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a concise 
summary of the following passage, covering the core pieces of 
information described.'
    
{article}
"""
class BasicLLMConfig(BaseModel):
    """
    Configuration of an LLM for summarization.
    """

    company: str
    model_name: str
    model_fullname: str | None = None # Model name if date_code is None or model name + date code otherwise.
    date_code: str | None = None  # some models have date codes, some don't. 
    prompt:str = default_prompt

    temperature: float = 0.0
    max_tokens: int = 4096
    min_throttle_time: float = 0.1  # number of seconds to wait before sending another request. Useful for web API calling that may have rate limits.
    thinking_tokens: int | None = None  # only applicable to models that support thinking.
    execution_mode: Literal["cpu", "gpu", "api"] | None = None # Call the LLM locally on GPU, on CPU), or through web API. Only applicable for open source models. 
    # interaction_mode: Literal["chat", "completion"] | None = None # When making a request, use the chat mode/endpoint or the completion mode/endpoint. Not applicable to all models. Almost all modern models do not distinguish between the two. 

    output_dir: str = "output"

    class Config:
        extra = "ignore"
    
    @classmethod
    def merge_with_defaults(cls, config_dict: dict, specific_config_class: type) -> dict:
        """
        Merge a config dictionary with defaults from a specific config class.
        
        Args:
            config_dict: Dictionary of config values
            specific_config_class: The specific config class to get defaults from
            
        Returns:
            dict: Merged config with defaults applied
        """
        # Get the default values from the specific config class
        default_config = specific_config_class()
        default_dict = default_config.model_dump()
        
        # Merge, with config_dict values taking precedence
        merged = default_dict.copy()
        for key, value in config_dict.items():
            if value is not None:  # Only override if value is not None
                merged[key] = value
        
        return merged

class BasicSummary(BaseModel):
    """
    Representation of a Summary of an Article
    
    Fields:
        eval_name (str): The name used to identify the evaluation (set in config.py as eval_name).
        eval_date (str): date of the evaluation
        summary_uid (str): hash for this summary
        company (str): company that produced the model
        model_name (str): name of the LLM used as summarizer
        date_code (str): date code of model, if applicable
        temperature (float): temperature of model
        max_tokens (int): max tokens allocated for model
        thinking_tokens(int): number of allocated thinking tokens
        article_id (int): unique id of article
        summary (str): llm generated summary of the text associated to article_id
    """
    eval_name: str
    eval_date: str
    article_id: int
    summary_uid: str
    summary: str

    company: str
    model_name: str
    date_code: str | None = None

    temperature: float
    max_tokens: int
    prompt: str
    thinking_tokens: int | None = None
    execution_mode: Literal["cpu", "gpu", "api"] | None = None

    model_config = {"extra": "ignore"}

    class Keys:
        EVAL_NAME = "eval_name"
        EVAL_DATE = "eval_date"
        SUMMARY_UID = "summary_uid"
        LLM = "llm"
        DATE_CODE = "date_code"
        TEMPERATURE = "temperature"
        MAX_TOKENS = "max_tokens"
        THINKING_TOKENS = "thinking_tokens"
        ARTICLE_ID = "article_id"
        SUMMARY = "summary"
class EvalConfig(BaseModel):
    """
    Configuration for an evaluation run. Includes the components of the evaluation pipeline, 
    the date of the evaluation, the LLMs to evaluate and their hyperparameters.

    Fields:
        eval_name (str): The name used to identify the evaluation
        eval_date (str): The date of the evaluation
        hhem_version: (str): Version of HHEM to use
        pipeline (List[Literal["summarize", "judge", "aggregate"]]): Steps to execute in the evaluation pipeline. Run sequentially. Meanings of the steps are as follows:
            - summarize: Generate summaries of the articles
            - judge: Judge the quality of the summaries
            - aggregate: Turn per-summary judgments into the hallucination rate, average summary length in words, and answer rate for the LLM, and the datecode if applicable, specified in the EvalConfig.
        overwrite_summaries (bool): if true overwrites all previously generated summaries. Only applicable to "summarize" step.
        source_article_path (str): path to file that contains the articles to be summarized by the LLMs
        temperature (float): the default temperature for this evaluation run, superseded by the temperature in the LLMConfig
        max_tokens (int): the maximum number of tokens for all LLMs, unless otherwise specified in the LLMConfig
        simulation_count (int): number of times a summary will be generated for the entire dataset
        sample_count (int): number of samples from simulations
        LLMs_to_eval (List[LLMConfig]): list of model configuration 
            representations
    """

    eval_name: str
    eval_date: str
    hhem_version: str
    pipeline: List[Literal["summarize", "judge", "aggregate"]]
    overwrite_summaries: bool # 
    output_dir: str
    source_article_path: str
    temperature: float = 0.0
    max_tokens: int = 1024
    simulation_count: int = 1  # no impact now 
    sample_count: int = 1      # no impact now 
    LLM_Configs: List[BasicLLMConfig]

    # TODO: Why is this function in the config? 
    def model_post_init(self, __context):
        for model_config in self.LLM_Configs:
            if model_config.temperature is None:
                model_config.temperature = self.temperature
            if model_config.max_tokens is None:
                model_config.max_tokens = self.max_tokens

    class Keys:
        PIPELINE = "pipeline"
        OVERWRITE = "overwrite"
        SOURCE_ARTICLE_PATH = "source_article_path"
        TEMPERATURE = "temperature"
        MAX_TOKENS = "max_tokens"
        SIMULATION_COUNT = "simulation_count"
        SAMPLE_COUNT = "sample_count"
        LLMS_TO_EVAL = "LLMs_to_eval"    
    
class SummaryError(str, Enum):
    MODEL_FAILED_TO_RETURN_OUTPUT = "MODEL FAILED TO RETURN ANY OUTPUT"
    MODEL_RETURNED_NON_STRING_TYPE_OUTPUT = (
        "DID NOT RECIEIVE A STRING TYPE FROM OUTPUT"
    )
    EMPTY_SUMMARY = (
        "THIS SUMMARY IS EMPTY, THIS IS THE DEFAULT VALUE A SUMMARY "
        "VARIABLE GETS. A REAL SUMMARY WAS NOT ASSIGNED TO THIS VARIABLE."
    )
    INCOMPLETE_THINK_TAG = "FOUND <think> WITH NO CLOSING </think>"

class ModelInstantiationError(str, Enum):
    CANNOT_EXECUTE_IN_MODE = "Model {model_name} by company {company} cannot execute in {execution_mode} mode."
    NOT_REGISTERED = "Model {model_name} by company {company} is not supported in HHEM Leaderboard. Please add a class under src.LLMs/ and update the registry."
    MISSING_SETUP = "Be sure to have a `setup` and a `teardown` method in the model class {class_name}. See `__enter__` and `__exit__` methods of `AbstractLLM` for more information."