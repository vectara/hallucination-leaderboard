from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
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


class BasicLLMConfig(BaseModel):
    """
    Configuration of an LLM for summarization.

    Please keep the default values of the optional fields as None. This ensures that when a user does not specify a value for an optional field, the field remains None. The non-none default values are set in the `AbstractLLM` class for all LLMs and the {Provider_name}LLM class under `src/LLMs/{Provider_name}.py` for LLMs from a specific provider. The default values of the provider-specific LLM class supercedes the default values of the provider-agnostic `AbstractLLM` class.
    """

    company: str
    model_name: str
    model_fullname: str | None = None # Model name if date_code is None or model name + date code otherwise.
    date_code: str | None = None  # some models have date codes, some don't. 
    prompt: str | None  = None

    temperature: float | None = None
    max_tokens: int | None = None
    min_throttle_time: float | None = None  # number of seconds to wait before sending another request. Useful for web API calling that may have rate limits.

    # Below are attributes that are not applicable to all models but are common in many models. We keep them here to establish a naming convention.
    thinking_tokens: int | None = None  # only applicable to models that support thinking.
    execution_mode: Literal["cpu", "gpu", "api"] | None = None # Call the LLM locally on GPU, on CPU), or through web API. Only applicable for open source models. 
    # interaction_mode: Literal["chat", "completion"] | None = None # When making a request, use the chat mode/endpoint or the completion mode/endpoint. Not applicable to all models. Almost all modern models do not distinguish between the two. 

    class Config:
        extra = "allow" # it must be allow because in `config.py`, there are LLM-specific parameters that are not in BasicLLMConfig.
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

    temperature: float | None = None
    max_tokens: int | None = None
    # prompt: str | None = None # We chose not to include prompt in the summary class because it is too long and normally not change. 
    thinking_tokens: int | None = None
    execution_mode: Literal["cpu", "gpu", "api"] | None = None

    class Config:
        extra = "ignore"
    class Keys:
        EVAL_NAME = "eval_name"
        EVAL_DATE = "eval_date"
        SUMMARY_UID = "summary_uid"
        
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

    """

    eval_name: str
    eval_date: str
    hhem_version: str
    pipeline: List[Literal["summarize", "judge", "aggregate"]]
    overwrite_summaries: bool # 
    source_article_path: str
    output_dir: str

    common_LLM_config: BasicLLMConfig
    per_LLM_configs: List[BasicLLMConfig]
    #TODO: Shall we call it config or params?

    # simulation_count: int = 1  # no impact now 
    # sample_count: int = 1      # no impact now

    class Config:
        extra = "ignore"

    # TODO: Why is this function in the config? 
    # def model_post_init(self, __context):
        # for model_config in self.LLM_Configs:
        #     if model_config.temperature is None:
        #         model_config.temperature = self.temperature
        #     if model_config.max_tokens is None:
        #         model_config.max_tokens = self.max_tokens

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
    CANNOT_EXECUTE_IN_MODE = "Model {model_name} by company {company} cannot execute in {execution_mode} mode. Because each LLM providers are different, please check the `setup()` method in class `{company}LLM` in `src/LLMs/{company}.py`."
    NOT_REGISTERED = "Model {model_name} by company {company} is not supported in HHEM Leaderboard. Please add a class under src.LLMs/ and update the registry."
    MISSING_SETUP = "Be sure to have a `setup` and a `teardown` method in the model class {class_name}. See `__enter__` and `__exit__` methods of `AbstractLLM` for more information."