import argparse
import os
from typing import List
import json

import pandas as pd
from dotenv import load_dotenv

from . data_model import EvalConfig, SourceArticle
from . Logger import logger
from . pipeline import (
    get_summaries, get_judgments, aggregate_judgments
)

def compile_results_for_all_llms(eval_config: EvalConfig) -> None:
    output_dir = eval_config.output_dir
    stats_jsonl = eval_config.stats_file

    columns = ["model_name", "date_code", "hallucination_rate", "confidence_interval", "answer_rate", "avg_word_count"] # The app only cares about these columns

    df_all_llms = pd.DataFrame(columns=columns) # init, empty dataframe

    # Compile all `output/{provider}/{llm_name}/stats.jsonl` files into one dataframe
    provider_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    for provider_name in provider_dirs:
        for llm_name in os.listdir(os.path.join(output_dir, provider_name)):
            stats_jsonl_path = os.path.join(output_dir, provider_name, llm_name, stats_jsonl)
            if os.path.isfile(stats_jsonl_path):
                df_provider = pd.read_json(stats_jsonl_path, lines=True)

                # For every df_provider, keep only one row that has the latest judgment_date (first) and summary_date (second) alphabetically
                df_provider = df_provider.sort_values(by=["judgment_date", "summary_date"], ascending=False)
                df_provider = df_provider.head(1)

                df_provider["model_name"] = df_provider["model_name"].apply(lambda x: f"{provider_name}/{x}")
                df_provider = df_provider[columns]
                df_all_llms = pd.concat([df_all_llms, df_provider]) 

    df_all_llms = df_all_llms.sort_values(by=["model_name"], ascending=True)
    output_path = os.path.join(output_dir, "stats_all_LLMs.json")
    with open(output_path, "w") as f:
        json.dump(
            df_all_llms.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2
        )
    # df_all_llms.to_json(os.path.join(output_dir, "stats_all_LLMs.json"), orient="records", indent=2)

# TODO: Move the main function to pipeline/__init__.py
def main(eval_config: EvalConfig) -> None:

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

        compile_results_for_all_llms(eval_config)
        # Todo: make it incremental. But may not be necessary because the compile function is very fast. 


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