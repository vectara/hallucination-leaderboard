"""Pydantic data models for the HHEM Leaderboard evaluation system.

This module defines the core data structures used throughout the hallucination
evaluation pipeline, including configuration models, record schemas, and
error enumerations.

Classes:
    SourceArticle: Schema for source document records from evaluation datasets.
    BasicJudgment: Schema for hallucination judgment results per summary.
    Stats: Schema for aggregated per-LLM statistics.
    BasicLLMConfig: Configuration model for LLM summarization settings.
    BasicSummary: Schema for generated summary records.
    EvalConfig: Configuration model for evaluation pipeline runs.
    SummaryError: Enumeration of summary generation error types.
    ModelInstantiationError: Enumeration of model setup error types.

Attributes:
    default_prompt: Default summarization prompt template.
"""

from typing import List, Literal, Dict, Any
from datetime import datetime

from enum import Enum
from pydantic import BaseModel, model_serializer

class SourceArticle(BaseModel):
    """Schema for source document records from evaluation datasets.

    Represents an article/document that will be summarized by LLMs and
    used as the ground truth for hallucination detection.

    Attributes:
        article_id: Unique identifier for the article.
        text: Full text content of the article.

    Example:
        >>> article = SourceArticle(article_id=1, text="Article content...")
    """

    article_id: int
    text: str
    # dataset: str

    class Keys:
        """Column name constants for DataFrame operations."""

        ARTICLE_ID = "article_id"
        TEXT = "text"
        # DATASET = "dataset"

class BasicJudgment(BaseModel):
    """Schema for hallucination judgment results per summary.

    Represents the evaluation metrics for a single summary, including the
    HHEM score indicating factual consistency with the source document.
    Provider-specific judgment classes may inherit from this base class.

    Attributes:
        eval_name: Name identifier for the evaluation run.
        judgment_date: ISO date string when the judgment was made.
        summary_uid: Unique hash identifier for the evaluated summary.
        hhem_version: Version string of the HHEM model used for scoring.
        hhem_score: Hallucination score from 0.0 to 1.0, where higher
            values indicate greater factual consistency.
        is_valid: Whether the summary passed validity checks.
        word_count: Number of words in the summary.

    Note:
        The date_code field is intentionally excluded from judgments as it
        is a property of the LLM configuration, not the summary itself.
    """

    eval_name: str
    judgment_date: str
    summary_uid: str
    hhem_version: str
    hhem_score: float
    is_valid: bool
    word_count: int

    class Keys:
        """Column name constants for DataFrame operations."""

        EVAL_NAME = "eval_name"
        EVAL_DATE = "eval_date"
        # DATE_CODE = "date_code"
        SUMMARY_UID = "summary_uid"
        HHEM_VERSION = "hhem_version"
        HHEM_SCORE = "hhem_score"
        IS_VALID = "is_valid"
        WORD_COUNT = "word_count"

class Stats(BaseModel):
    """Schema for aggregated per-LLM statistics.

    Represents aggregate metrics computed from individual summary judgments
    for a single LLM under a specific evaluation configuration. Used for
    leaderboard rankings and model comparison.

    Attributes:
        eval_name: Name identifier for the evaluation run.
        summary_date: ISO date string when summaries were generated.
        judgment_date: ISO date string when judgments were made.
        model_name: Name of the evaluated LLM.
        date_code: Optional version/date identifier for the model.
        hhem_version: Version string of the HHEM model used.
        hallucination_rate: Percentage of summaries classified as hallucinated.
        confidence_interval: Statistical confidence interval for the rate.
        answer_rate: Percentage of summaries that passed validity checks.
        avg_word_count: Mean word count across valid summaries.
    """

    eval_name: str
    summary_date: str
    judgment_date: str
    model_name: str
    date_code: str | None = None
    hhem_version: str

    hallucination_rate: float
    confidence_interval: float
    answer_rate: float
    avg_word_count: float

    class Config:
        """Pydantic configuration allowing extra fields for LLM settings."""

        extra = "allow"  # put all settings for LLM generation here, including prompt

    class Keys:
        """Column name constants for DataFrame operations."""

        EVAL_NAME = "eval_name"
        EVAL_DATE = "eval_date"
        MODEL_NAME = "model_name"
        DATE_CODE = "date_code"
        HHEM_VERSION = "hhem_version"
        HALLUCINATION_RATE = "hallucination_rate"
        CONFIDENCE_INTERVAL = "confidence_interval"
        ANSWER_RATE = "answer_rate"
        AVG_WORD_COUNT = "avg_word_count"

default_prompt = """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the
passage provided. You are asked the question 'Provide a concise
summary of the following passage, covering the core pieces of
information described.'

{article}
"""
"""str: Default summarization prompt template with {article} placeholder."""


class BasicLLMConfig(BaseModel):
    """Configuration model for LLM summarization settings.

    Base configuration class for all LLM providers. Provider-specific
    config classes inherit from this and add additional fields as needed.
    Supports both API-based and local model execution.

    Attributes:
        company: Provider identifier (e.g., "openai", "anthropic").
        model_name: Name of the specific model variant.
        date_code: Optional version/date identifier for model snapshots.
        prompt: Summarization prompt template with {article} placeholder.
        threads: Number of concurrent threads for summary generation.
        temperature: Sampling temperature for generation (0.0 = deterministic).
        max_tokens: Maximum tokens in generated summary.
        min_throttle_time: Minimum seconds between API requests for rate limiting.
        thinking_tokens: Token budget for reasoning models (if applicable).
        execution_mode: Runtime environment ("api", "cpu", or "gpu").
    """

    company: str = "ANYCOMPANY"
    model_name: str = "ANYMODEL"
    # model_fullname: str | None = None # Model name if date_code is None or model name + date code otherwise.
    date_code: str = ""  # some models have date codes, some don't. 
    prompt: str = default_prompt
    threads: int = 1

    temperature: float = 0.0
    max_tokens: int = 4096
    min_throttle_time: float = 2.0  # number of seconds to wait before sending another request. Useful for web API calling that may have rate limits.

    # Below are attributes that are not applicable to all models but are common in many models. We keep them here to establish a naming convention.
    thinking_tokens: int | None = None  # Number of tokens allocated for thinking. Only applicable to models that support thinking.
    execution_mode: Literal["cpu", "gpu", "api"] | None = None # Call the LLM locally on GPU, on CPU), or through web API. Only applicable for open source models. 
    # interaction_mode: Literal["chat", "completion"] | None = None # When making a request, use the chat mode/endpoint or the completion mode/endpoint. Not applicable to all models. Almost all modern models do not distinguish between the two. 

    @property
    def model_fullname(self) -> str:
        """Build the full model identifier from name and date_code.

        Combines model_name and date_code into a single identifier string.
        Individual LLM classes may override this with custom formatting.

        Returns:
            Model name if date_code is None/empty, otherwise "model_name-date_code".
        """
        if self.date_code is None:
            return self.model_name
        return f"{self.model_name}-{self.date_code}"

    @model_serializer
    def clean_model_dump(self):
        """Serialize model excluding internal/transient fields.

        Returns:
            Dictionary of model fields excluding min_throttle_time and
            model_fullname which are not needed in serialized output.
        """
        fields_to_exclude = ['min_throttle_time', 'model_fullname']
        return {k: v for k, v in self.__dict__.items() if k not in fields_to_exclude}

    class Config:
        """Pydantic configuration for LLM config validation."""

        extra = "ignore"  # TODO: maybe we shall set to forbid to warn users of extra fields.
        validate_assignment = True  # Always validate after updating
        # valide_assignment == True may be a problem for some cases such as the one here https://stackoverflow.com/questions/62025723/how-to-validate-a-pydantic-object-after-editing-it#comment132889958_62027169 but not a problem for us because all configurable parameters are indenpendent. 

class BasicSummary(BaseModel):
    """Schema for generated summary records.

    Base class for LLM-generated summaries. Provider-specific summary classes
    inherit from this and may add additional fields for tracking provider-
    specific metadata (e.g., endpoint type, reasoning mode).

    Attributes:
        article_id: ID of the source article that was summarized.
        summary_uid: Unique hash identifier for this summary.
        summary: The generated summary text content.
        company: Provider identifier for the generating LLM.
        model_name: Name of the model that generated the summary.
        date_code: Optional version/date identifier for the model.
        eval_name: Name identifier for the evaluation run.
        summary_date: ISO date string when the summary was generated.
        temperature: Sampling temperature used for generation.
        max_tokens: Maximum token limit used for generation.
        thinking_tokens: Reasoning tokens used (for reasoning models).
        execution_mode: Runtime environment used ("api", "cpu", or "gpu").

    Note:
        The prompt field is intentionally excluded as it is typically
        lengthy and does not vary between summaries in an evaluation run.
    """

    article_id: int
    summary_uid: str
    summary: str

    company: str
    model_name: str
    date_code: str | None = None

    eval_name: str
    summary_date: str

    temperature: float | None = None
    max_tokens: int | None = None
    # prompt: str | None = None # We chose not to include prompt in the summary class because it is too long and normally not change.
    thinking_tokens: int | None = None
    execution_mode: Literal["cpu", "gpu", "api"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"  # TODO: maybe we shall set to forbid to warn users of extra fields.

    class Keys:
        """Column name constants for DataFrame operations."""

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
    """Configuration model for evaluation pipeline runs.

    Defines the complete configuration for an evaluation run including
    pipeline stages to execute, LLM configurations, input/output paths,
    and HHEM model version.

    Attributes:
        eval_name: Unique name identifier for this evaluation run.
        eval_date: ISO date string for the evaluation.
        hhem_version: Version of HHEM model to use for judging.
        pipeline: List of pipeline stages to execute in order.
        overwrite_summaries: Whether to overwrite existing summary files.
        source_article_path: Path to CSV file containing source articles.
        output_dir: Base directory for output files.
        common_LLM_config: Shared configuration applied to all LLMs.
        per_LLM_configs: List of model-specific configurations.
        summary_file: Filename for summary JSONL output.
        judgment_file: Filename for judgment JSONL output.
        stats_file: Filename for statistics JSONL output.

    Example:
        >>> config = EvalConfig(
        ...     eval_name="test_run",
        ...     eval_date="2024-01-15",
        ...     hhem_version="2.3",
        ...     common_LLM_config=BasicLLMConfig(),
        ...     per_LLM_configs=[openai_config, anthropic_config]
        ... )
    """

    eval_name: str
    eval_date: str
    hhem_version: str
    pipeline: List[Literal["summarize", "judge", "aggregate", "compile_results"]] = ["summarize", "judge", "aggregate", "compile_results"]
    overwrite_summaries: bool = False
    source_article_path: str = "datasets/test_articles.csv"
    output_dir: str = "./output"

    common_LLM_config: BasicLLMConfig
    per_LLM_configs: List[BasicLLMConfig]

    # Default output files for pipeline steps
    summary_file: str = "summaries.jsonl"
    judgment_file: str = "judgments.jsonl"
    stats_file: str = "stats.jsonl"

    # simulation_count: int = 1  # no impact now
    # sample_count: int = 1      # no impact now

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"  # TODO: maybe we shall set to forbid to warn users of extra fields.

    class Keys:
        """Column name constants for configuration access."""

        PIPELINE = "pipeline"
        OVERWRITE = "overwrite"
        SOURCE_ARTICLE_PATH = "source_article_path"
        TEMPERATURE = "temperature"
        MAX_TOKENS = "max_tokens"
        SIMULATION_COUNT = "simulation_count"
        SAMPLE_COUNT = "sample_count"
        LLMS_TO_EVAL = "LLMs_to_eval"    
    
class SummaryError(str, Enum):
    """Enumeration of summary generation error types.

    String enum values used as placeholder text when summary generation
    fails. These values are checked by is_valid_summary() to identify
    failed generations.

    Attributes:
        MODEL_FAILED_TO_RETURN_OUTPUT: Model API call returned no response.
        MODEL_RETURNED_NON_STRING_TYPE_OUTPUT: Response was not a string type.
        EMPTY_SUMMARY: Default placeholder indicating no summary was assigned.
        INCOMPLETE_THINK_TAG: Reasoning tags were not properly closed.
        SUMMARY_REFUSAL_OUTPUT: Model refused to summarize the content.
        GIVEN_EMPTY_SUMMARY: Model returned an empty string.
        THREAD_ERROR: Multi-threaded worker encountered an error.
    """

    MODEL_FAILED_TO_RETURN_OUTPUT = "MODEL FAILED TO RETURN ANY OUTPUT"
    MODEL_RETURNED_NON_STRING_TYPE_OUTPUT = (
        "DID NOT RECIEIVE A STRING TYPE FROM OUTPUT"
    )
    EMPTY_SUMMARY = (
        "THIS SUMMARY IS EMPTY, THIS IS THE DEFAULT VALUE A SUMMARY "
        "VARIABLE GETS. A REAL SUMMARY WAS NOT ASSIGNED TO THIS VARIABLE."
    )
    INCOMPLETE_THINK_TAG = "FOUND <think> WITH NO CLOSING </think>"
    SUMMARY_REFUSAL_OUTPUT = "I am unable to summarize this passage."
    GIVEN_EMPTY_SUMMARY = "EMPTY SUMMARY GIVEN BY MODEL"
    THREAD_ERROR = "THREAD ERROR"


class ModelInstantiationError(str, Enum):
    """Enumeration of model setup and instantiation error types.

    String enum values with format placeholders for constructing detailed
    error messages when LLM initialization fails.

    Attributes:
        CANNOT_EXECUTE_IN_MODE: Model does not support the requested
            execution mode (api/cpu/gpu).
        NOT_REGISTERED: Model/company combination is not in the registry.
        MISSING_SETUP: LLM class is missing required setup/teardown methods.
    """

    CANNOT_EXECUTE_IN_MODE = "Model {model_name} by company {company} cannot execute in {execution_mode} mode. Because each LLM providers are different, please check the `setup()` method in class `{company}LLM` in `src/LLMs/{company}.py`."
    NOT_REGISTERED = "Model {model_name} by company {company} is not supported in HHEM Leaderboard. Please add a class under src.LLMs/ and update the registry."
    MISSING_SETUP = "Be sure to have a `setup` and a `teardown` method in the model class {class_name}. See `__enter__` and `__exit__` methods of `AbstractLLM` for more information."