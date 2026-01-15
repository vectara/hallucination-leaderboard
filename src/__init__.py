"""HHEM Leaderboard Backend for LLM hallucination evaluation.

A comprehensive system for evaluating Large Language Model (LLM) hallucination
using the Hughes Hallucination Evaluation Model (HHEM). Provides an end-to-end
pipeline for generating summaries, detecting hallucinations, and computing
performance metrics across multiple LLM providers.

Modules:
    LLMs: LLM provider integrations (OpenAI, Anthropic, Google, Meta, etc.).
    pipeline: Core evaluation pipeline (summarize, judge, aggregate).
    data_model: Pydantic models for configuration and data structures.
    analytics: Statistical computation and metrics functions.
    HHEM_2_x: HHEM model implementations for hallucination detection.
    json_utils: Utilities for JSONL file operations.
    Logger: Centralized logging configuration.

Key Features:
    - Multi-provider LLM support with unified interface
    - Configurable summarization with single/multi-threaded execution
    - Hallucination scoring via HHEM model family
    - Statistical aggregation with confidence intervals
    - Incremental result persistence to JSONL files

Example:
    >>> from src.pipeline import get_summaries, get_judgments, aggregate_judgments
    >>> from src.data_model import EvalConfig
    >>> config = EvalConfig.from_yaml("config.yaml")
    >>> get_summaries(config, article_df)
    >>> get_judgments(config, article_df)
    >>> aggregate_judgments(config)

Attributes:
    __version__: Package version string.
    __author__: Package author name.
"""

__version__ = "0.1.0"
"""str: Current version of the HHEM Leaderboard package."""

__author__ = "Forrest"
"""str: Primary author of the package."""