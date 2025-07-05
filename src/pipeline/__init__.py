"""
Scripts Package

This package contains the main pipeline scripts for:
- Generating summaries from LLMs
- Computing judgements and metrics
- Generating final results and statistics
"""

from . get_summaries import run as get_summaries
from . get_judgements import run as get_judgements
# from . get_results import run as get_results

# Expose the modules themselves for nested imports
# from . import get_summaries
# from . import get_judgements
# from . import get_results
# from . import get_hallucination_rates

# __all__ = [
#     "get_summaries",
#     "get_judgements", 
#     "get_results",
#     "get_hallucination_rates",
# ] 