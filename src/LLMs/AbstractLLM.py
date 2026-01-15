"""Abstract base class for LLM implementations.

This module defines the AbstractLLM base class that all LLM provider
implementations must inherit from. It provides a unified interface for
text summarization across different model providers and execution modes.

The AbstractLLM class supports:
    - Context manager protocol for resource management (setup/teardown).
    - Both API client and local model inference modes.
    - Automatic thinking tag removal for reasoning models.
    - Rate limiting via configurable throttle times.

Classes:
    AbstractLLM: Abstract base class defining the LLM interface.

Example:
    >>> class MyLLM(AbstractLLM):
    ...     def setup(self): ...
    ...     def teardown(self): ...
    ...     def summarize(self, text): ...
    ...     def close_client(self): ...
    >>> config = MyConfig(company="myco", model_name="my-model")
    >>> with MyLLM(config) as llm:
    ...     summary = llm.try_to_summarize_one_article("Article text...")
"""

import os
import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Literal, Any
import pandas as pd

import torch
from pydantic import BaseModel
from tqdm import tqdm

from .. data_model import ModelInstantiationError, BasicLLMConfig, SummaryError
from .. Logger import logger

# Definitions below moved to constants.py -- Forrest, 2025-07-02
# MODEL_FAILED_TO_RETURN_OUTPUT = "MODEL FAILED TO RETURN ANY OUTPUT"
# MODEL_RETURNED_NON_STRING_TYPE_OUTPUT = (
#     "DID NOT RECEIVE A STRING TYPE FROM OUTPUT"
# )
# EMPTY_SUMMARY = (
#     "THIS SUMMARY IS EMPTY, THIS IS THE DEFAULT VALUE A SUMMARY "
#     "VARIABLE GETS. A REAL SUMMARY WAS NOT ASSIGNED TO THIS VARIABLE."
# )
# INCOMPLETE_THINK_TAG = "FOUND <think> WITH NO CLOSING </think>"

# SUMMARY_ERRORS = [
#     MODEL_FAILED_TO_RETURN_OUTPUT,
#     MODEL_RETURNED_NON_STRING_TYPE_OUTPUT,
#     EMPTY_SUMMARY,
#     INCOMPLETE_THINK_TAG
# ]

class AbstractLLM(ABC):
    """Abstract base class for all LLM provider implementations.

    Defines the interface that all LLM implementations must follow, providing
    common functionality for text summarization, resource management, and
    error handling. Subclasses must implement the abstract methods: setup(),
    teardown(), summarize(), and close_client().

    Supports both API-based inference (via self.client) and local model
    inference (via self.local_model). The class implements the context manager
    protocol for automatic resource cleanup.

    Attributes:
        company: Provider/company identifier string.
        model_name: Name of the specific model variant.
        prompt: Template string for formatting input text.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens to generate in response.
        min_throttle_time: Minimum seconds between API calls for rate limiting.
        date_code: Optional version/date identifier for the model.
        thinking_tokens: Configuration for reasoning/thinking tokens.
        execution_mode: Where to run inference ("api", "gpu", "cpu").
        model_fullname: Complete model identifier including date code.
        client: API client instance for remote inference, if applicable.
        local_model: Local model instance for on-device inference, if applicable.
        device: PyTorch device (cuda or cpu) for local inference.
    """

    def __init__(self, config: BasicLLMConfig) -> None:
        """Initialize the LLM with configuration settings.

        Args:
            config: Configuration object containing model settings and parameters.
        """
        # Expose all config keys and values as attributes on self
        # for key, value in config.model_dump().items():
        #     setattr(self, key, value)

        self.company = config.company
        self.model_name = config.model_name

        # Set defaults for optional attributes
        # self.prompt = config.prompt if config.prompt is not None else default_prompt
        # self.temperature = config.temperature if config.temperature is not None else 0.0
        # self.max_tokens = config.max_tokens if config.max_tokens is not None else 1024
        # self.min_throttle_time = config.min_throttle_time if config.min_throttle_time is not None else 0.1

        self.prompt = config.prompt
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.min_throttle_time = config.min_throttle_time

        # The following attributes are not required by all models.
        self.date_code = config.date_code       
        self.thinking_tokens = config.thinking_tokens
        self.execution_mode = config.execution_mode

        if self.date_code not in [None, "", " "]:
            self.model_fullname = f"{self.model_name}-{self.date_code}"
        else:
            self.model_fullname = self.model_name

        self.client: Any | None = None # in case the model can be called via web api
        self.local_model: Any | None = None # in case the model can be run locally
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.summary_file: str | None = None # won't be set until after instantiation in summarize.py

    def __enter__(self):
        """Enter the context manager, initializing the model.

        Calls setup() to prepare the model for inference.

        Returns:
            self: The initialized LLM instance.
        """
        self.setup() # TODO: Try to skip the setup() and teardown()
        return self

    def __exit__(self, exc_type, exc_val, exc_t):
        """Exit the context manager, cleaning up resources.

        Calls teardown() to release model resources regardless of whether
        an exception occurred.

        Args:
            exc_type: Exception type if an exception was raised, else None.
            exc_val: Exception value if an exception was raised, else None.
            exc_t: Exception traceback if an exception was raised, else None.
        """
        self.teardown()

    # Commented out by Forrest, 2025-07-16 
    # Due to not used at all 
    # def summarize_articles(self, articles: list[str]) -> list[str]:
    #     """
    #     Takes in a list of articles, iterates through the list. Returns list of
    #     the summaries

    #     Args:
    #         articles (list[str]): List of strings where the strings are human
    #         written news articles

    #     Returns:
    #         list[str]: List of articles generated by the LLM
    #     """
    #     summaries = []
    #     for article in tqdm(articles, desc="Article Loop"):
    #         summary = self.summarize_clean_wait(article)
    #         summaries.append(summary)
    #     return summaries

    def try_to_summarize_one_article(self, article: str) -> str:
        """Attempt to summarize an article with error handling.

        Formats the article with the configured prompt template, requests a
        summary from the model, and handles any errors gracefully. Applies
        rate limiting based on min_throttle_time and removes thinking tags
        from the response.

        Args:
            article: The article text to be summarized.

        Returns:
            The generated summary text, or an error message string if
            summarization fails (e.g., MODEL_FAILED_TO_RETURN_OUTPUT,
            MODEL_RETURNED_NON_STRING_TYPE_OUTPUT).
        """

        prepared_llm_input = self.prompt.format(article=article)

        start_time = time.time()

        try:
            llm_summary = self.summarize(prepared_llm_input)
        except Exception as e:
            logger.warning((
                f"Model call failed for {self.model_name}: {e} "
            ))
            return SummaryError.MODEL_FAILED_TO_RETURN_OUTPUT

        if not isinstance(llm_summary, str):
            bad_output = llm_summary
            logger.warning((
                f"{self.model_name} returned unexpected output. Expected a "
                f"string but got {type(bad_output).__name__}. "
                f"Replacing output."
            ))
            return SummaryError.MODEL_RETURNED_NON_STRING_TYPE_OUTPUT
        
        llm_summary = self.remove_thinking_text(llm_summary)

        elapsed_time = time.time() - start_time
        remaining_time = self.min_throttle_time - elapsed_time
        if remaining_time > 0:
            time.sleep(remaining_time)

        if llm_summary == "":
            llm_summary = "EMPTY SUMMARY GIVEN BY MODEL"

        return llm_summary

    def remove_thinking_text(self, raw_summary: str) -> str:
        """Remove thinking/reasoning tags and their content from model output.

        Strips <think>...</think> blocks from the summary, which are used by
        some reasoning models to show their thought process. If an opening
        <think> tag is found without a matching </think>, the response is
        considered incomplete and an error string is returned.

        Note:
            Different LLMs may use different tags for thinking. Consider
            overriding in subclasses if a model uses non-standard tags.

        Args:
            raw_summary: The raw summary text from the LLM, potentially
                containing thinking tags.

        Returns:
            The summary with thinking content removed, or
            INCOMPLETE_THINK_TAG error string if the response is malformed.
        """
        if '<think>' in raw_summary and '</think>' not in raw_summary:
            logger.warning(f"<think> tag found with no </think>. This is indicative of an incomplete response from an LLM. Raw Summary: {raw_summary}")
            return SummaryError.INCOMPLETE_THINK_TAG

        summary = re.sub(
            r'<think>.*?</think>\s*', '',
            raw_summary, flags=re.DOTALL
        )
        return summary

    def default_local_model_teardown(self):
        """Standard teardown protocol for PyTorch-based local models.

        Deletes the local model, clears the CUDA cache to free GPU memory,
        and resets the local_model attribute to None. Call this method from
        subclass teardown() implementations when using local inference.
        """
        # self.local_model.to("cpu")
        del self.local_model
        torch.cuda.empty_cache()
        self.local_model = None

    def prepare_for_overwrite(self, summaries_jsonl_path: str, summary_date: str):
        """Prepare the summary file for overwriting with new results.

        Clears the existing summary file to allow fresh results to be written.
        Used when re-running summarization for a model.

        Args:
            summaries_jsonl_path: Path to the JSONL file storing summaries.
            summary_date: Date identifier for the summarization run (unused
                in current implementation but reserved for future filtering).
        """
        open(summaries_jsonl_path, 'w').close()


        # TODO: Unsure the need for this logic
        # if self.date_code in [None, "", " "]:
        #     # Clean the summary file 
        #     open(summaries_jsonl_path, 'w').close()
        # else: 
        #     # Remove summaries in existing summary file that match the model name, date code, and summary_date
        #     df = pd.read_json(summaries_jsonl_path, lines=True)
        #     df = df[(df['model_name'] != self.model_name) | (df['date_code'] != self.date_code) | (df['summary_date'] != summary_date)]
        #     df.to_json(summaries_jsonl_path, orient='records', lines=True)

    @abstractmethod
    def summarize(self, prepared_text: str) -> str:
        """Generate a summary from the prepared input text.

        This is the core inference method that subclasses must implement.
        It should invoke the model (via API client or local inference) to
        generate a summary of the provided text.

        Args:
            prepared_text: The prompt-formatted text ready for model input.

        Returns:
            The generated summary text from the model.
        """
        return None

    @abstractmethod
    def setup(self):
        """Initialize the model and prepare for inference.

        Subclasses must implement this to set up either self.client (for API
        inference) or self.local_model (for local inference). Called automatically
        when entering a context manager block.
        """
        return None

    @abstractmethod
    def teardown(self):
        """Release resources and clean up after inference.

        Subclasses must implement this to properly release model resources.
        For local models, consider calling default_local_model_teardown().
        Called automatically when exiting a context manager block.
        """
        return None

    @abstractmethod
    def close_client(self):
        """Close any active API client connections.

        Lately it is not necessary to do this, this method may be removed in
        the future.
        """
        return None