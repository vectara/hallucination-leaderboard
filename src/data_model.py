from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum

from . LLMs import LLMConfig

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

class BasicSummary(BaseModel):
    """
    Representation of a Summary of an Article
    
    Fields:
        timestamp (str): Date summary was produced
        summary_uid (str): hash for this summary
        llm (str): unique llm identifier, matches the label the respective
            company gave it
        date_code (str): date code of model
        temperature (float): temperature of model
        max_tokens (int): max tokens allocated for model
        thinking_tokens(int): number of allocated thinking tokens
        article_id (int): unique id of article
        summary (str): llm generated summary of the text associated to article_id
    """
    timestamp: str
    summary_uid: str
    llm: str
    date_code: str
    interaction_mode: str
    temperature: float
    max_tokens: int
    thinking_tokens: int
    article_id: int
    summary: str

    model_config = {"extra": "allow"}

    class Keys:
        TIMESTAMP = "timestamp"
        SUMMARY_UID = "summary_uid"
        LLM = "llm"
        DATE_CODE = "date_code"
        TEMPERATURE = "temperature"
        MAX_TOKENS = "max_tokens"
        THINKING_TOKENS = "thinking_tokens"
        ARTICLE_ID = "article_id"
        SUMMARY = "summary"

class Judgement(BaseModel):
    """
    Representation of Judgements/Metrics for the Summary of an Article

    Fields:
        timestamp (str): date the metrics were performed
        summary_uid (str): hash for this summary
        date_code (str): date code of model
        hhem_version (str): version of hhem applied for hhem score
        hhem_score (float): Hughes Hallucination Evaluation Metric (HHEM)
        valid (bool): Validity of the summary, defined in is_valid_summary
        summary_words (int): word count of summary
    """
    timestamp: str
    summary_uid: str
    date_code: str
    hhem_version: str
    hhem_score: float
    valid: bool
    summary_words: int

    class Keys:
        TIMESTAMP = "timestamp"
        SUMMARY_UID = "summary_uid"
        DATE_CODE = "date_code"
        HHEM_VERSION = "hhem_version"
        HHEM_SCORE = "hhem_score"
        VALID = "valid"
        SUMMARY_WORDS = "summary_words"

class Stats(BaseModel):
    """
    Representation of Stats for the Summaries of the Article Dataset. These are
    aggregated Judgements/Metrics

    Fields:
        timestamp (str): date the stats were performed
        llm (str): llm that performed the summarization
        date_code (str): date code of model
        hallucination_rate (float): hallucination rate on all summaries
        confidence_interval (float): variation in the hallucination rate
        answer_rate (float): For all summaries what percentage of them were
            valid
        avg_summary_length (float): Average summary length for all valid
            summaries
    """
    timestamp: str
    llm: str
    date_code: str
    hallucination_rate: float
    confidence_interval: float
    answer_rate: float
    avg_summary_length: float

    class Keys:
        TIMESTAMP = "timestamp"
        LLM = "llm"
        DATE_CODE = "date_code"
        HALLUCINATION_RATE = "hallucination_rate"
        CONFIDENCE_INTERVAL = "confidence_interval"
        ANSWER_RATE = "answer_rate"
        AVG_SUMMARY_LENGTH = "avg_summary_length"

class EvalConfig(BaseModel):
    """
    Configuration for an evaluation run. Includes the components of the evaluation pipeline, 
    the date of the evaluation, the LLMs to evaluate and their hyperparameters.

    Fields:
        eval_date: str # Date of the evaluation
        hhem_version: str # Version of HHEM to use
        pipeline: List[Literal["summarize", "judge", "reduce"]] 
                  Steps to execute in the evaluation pipeline. Run sequentially.
                  Meanings of the steps are as follows:
                  - summarize: Generate summaries of the articles
                  - judge: Judge the quality of the summaries
                  - reduce: Turn per-summary judgements into the hallucination rate, average summary length in words, and answer rate for each LLM.
        overwrite_summaries (bool): if true overwrites all previously generated summaries. Only applicable to "summarize" step.
        source_article_path (str): path to file that contains the articles to be summarized by the LLMs
        temperature (float): the default temperature for this evaluation run, superseded by the temperature in the LLMConfig
        max_tokens (int): the maximum number of tokens for all LLMs, unless otherwise specified in the LLMConfig
        simulation_count (int): number of times a summary will be generated for the entire dataset
        sample_count (int): number of samples from simulations
        LLMs_to_eval (List[LLMConfig]): list of model configuration 
            representations
    """

    eval_date: str
    hhem_version: str
    pipeline: List[Literal["summarize", "judge", "reduce"]]
    overwrite_summaries: bool # 
    source_article_path: str
    temperature: float = 0.0
    max_tokens: int = 1024
    simulation_count: int = 1  # no impact now 
    sample_count: int = 1      # no impact now 
    LLM_Configs: List[LLMConfig]

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