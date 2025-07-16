import argparse
from typing import List

import pandas as pd
from dotenv import load_dotenv

from . data_model import EvalConfig, SourceArticle
from . Logger import logger
from . pipeline import (
    get_summaries, get_judgments, aggregate_judgments
)

# TODO: Move the main function to pipeline/__init__.py
def main(eval_config: EvalConfig):

    article_df = pd.read_csv(eval_config.source_article_path)
    
    # Type check: ensure every row is a valid SourceArticle
    try:
        # Convert DataFrame to list of dicts and validate all at once
        articles_data = article_df.to_dict('records')
        validated_articles = [SourceArticle.model_validate(article) for article in articles_data]
        logger.info(f"Successfully validated {len(validated_articles)} SourceArticle objects")
    except Exception as e:
        raise ValueError(f"Source article data validation failed: {e}")

    if "summarize" in eval_config.pipeline:
        get_summaries(eval_config, article_df)

    if "judge" in eval_config.pipeline:
        get_judgments(eval_config, article_df)
    
    if "aggregate" in eval_config.pipeline:
        aggregate_judgments(eval_config)

def cli_main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="HHEM Leaderboard Backend",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Load config.py
    from . config import eval_configs

    parser.add_argument(
        "--eval_name",
        choices=[eval_config.eval_name for eval_config in eval_configs],
        help="Which evaluation to run. Pick evaluation name from config.py. For test, select 'test'.",
        default="test"
    )

    args = parser.parse_args()
    eval_name = args.eval_name
    logger.info(f"Running evaluation {eval_name}")

    # Qualify the eval_name to get the correct eval_config
    qualified_eval_config =  [eval_config for eval_config in eval_configs if eval_config.eval_name == eval_name]
    if len(qualified_eval_config) == 0:
        raise ValueError(f"Evaluation {eval_name} not found in config.py")
    elif len(qualified_eval_config) > 1:
        raise ValueError(f"Evaluation {eval_name} found multiple times in config.py")
    else:
        eval_config = qualified_eval_config[0]

    main(eval_config)

if __name__ == "__main__":
    cli_main()