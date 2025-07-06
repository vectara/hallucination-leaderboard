"""
Scripts Package

This package contains the main pipeline scripts for:
- Generating summaries from LLMs
- Judge each summary
- Generating final results and statistics
"""

from . summarize import get_summaries
from . judge import get_judgments
from . aggregate import aggregate_judgments
