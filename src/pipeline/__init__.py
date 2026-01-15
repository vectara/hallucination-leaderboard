"""Pipeline package for hallucination evaluation workflows.

This package contains the main pipeline modules for running end-to-end
hallucination evaluation experiments. The pipeline consists of three
sequential stages: summarization, judgment, and aggregation.

Modules:
    summarize: Generate summaries from LLMs given source documents.
    judge: Evaluate generated summaries for factual consistency.
    aggregate: Compile judgment results into final statistics.

Functions:
    get_summaries: Generate summaries using configured LLM models.
    get_judgments: Evaluate summaries against source documents.
    aggregate_judgments: Aggregate judgment results into statistics.

Example:
    >>> from src.pipeline import get_summaries, get_judgments, aggregate_judgments
    >>> summaries = get_summaries(config)
    >>> judgments = get_judgments(summaries)
    >>> results = aggregate_judgments(judgments)
"""

from . summarize import get_summaries
from . judge import get_judgments
from . aggregate import aggregate_judgments
