"""Main entry point for the HHEM Leaderboard evaluation system.

This module provides the command-line interface and orchestration logic
for running hallucination evaluation pipelines. Coordinates the three
main pipeline stages (summarize, judge, aggregate) and compiles results
across all evaluated LLMs.

Functions:
    compile_results_for_all_llms: Aggregate stats from all LLMs into one file.
    main: Execute the evaluation pipeline based on configuration.
    cli_main: Command-line interface entry point.

Example:
    Command-line usage::

        python -m src.main --eval_name test
        python -m src.main --eval_name production

    Programmatic usage::

        from src.main import main
        from src.data_model import EvalConfig
        config = EvalConfig(...)
        main(config)
"""

import argparse
import json
import os
from typing import List

import pandas as pd
from dotenv import load_dotenv

from . data_model import EvalConfig, SourceArticle
from . Logger import logger
from . pipeline import (
    get_summaries, get_judgments, aggregate_judgments
)


def compile_results_for_all_llms(eval_config: EvalConfig) -> None:
    """Compile statistics from all evaluated LLMs into a single JSON file.

    Traverses the output directory structure, reads per-LLM stats files,
    and aggregates them into a unified JSON file for the leaderboard
    frontend. For each LLM, only the most recent results (by judgment
    and summary date) are included.

    Args:
        eval_config: Evaluation configuration containing output directory
            path and stats file naming convention.

    Output:
        Creates "{output_dir}/stats_all_LLMs.json" containing an array of
        objects with fields: model_name, date_code, hallucination_rate,
        confidence_interval, answer_rate, avg_word_count.

    Note:
        Model names in the output are prefixed with the provider name
        (e.g., "openai/gpt-4o") and suffixed with date_code if present.
        Empty stats files are skipped with a warning message.
    """
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
                df_provider = pd.read_json(stats_jsonl_path, lines=True, dtype={"date_code": str})

                if df_provider.empty:
                    print(f"Stats file {stats_jsonl_path} is empty. Skipping {provider_name}/{llm_name}.")
                    continue

                # For every df_provider, keep only one row that has the latest judgment_date (first) and summary_date (second) alphabetically
                df_provider = df_provider.sort_values(by=["judgment_date", "summary_date"], ascending=False)
                df_provider = df_provider.head(1)

                # df_provider["model_name"] = df_provider["model_name"].apply(lambda x: f"{provider_name}/{x}")
                def model_alias(row):
                    base = f"{provider_name}/{row['model_name']}"
                    if 'date_code' in row and isinstance(row['date_code'], str) and row['date_code'].strip():
                        return f"{base}-{row['date_code']}"
                    return base
                df_provider["model_name"] = df_provider.apply(model_alias, axis=1)

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
    """Execute the evaluation pipeline based on configuration.

    Main orchestration function that runs the configured pipeline stages
    in sequence. Loads source articles, validates them against the
    SourceArticle schema, and executes each enabled stage.

    Pipeline stages (controlled by eval_config.pipeline):
        - "summarize": Generate summaries using configured LLMs.
        - "judge": Score summaries using HHEM for hallucination detection.
        - "aggregate": Compute per-LLM statistics from individual judgments.
        - "compile_results": Merge all LLM stats into a single output file.

    Args:
        eval_config: Complete evaluation configuration including pipeline
            stages to run, LLM configurations, file paths, and settings.

    Raises:
        ValueError: If source article data fails schema validation.

    Note:
        The "aggregate" stage automatically triggers "compile_results"
        for convenience. The "compile_results" stage can also be run
        independently to regenerate the combined stats file.
    """
    article_df = pd.read_csv(eval_config.source_article_path)
    article_df = article_df[[SourceArticle.Keys.ARTICLE_ID, SourceArticle.Keys.TEXT]]
    
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

    if "compile_results" in eval_config.pipeline:
        compile_results_for_all_llms(eval_config)

        # Todo: make it incremental. But may not be necessary because the compile function is very fast. 


def cli_main():
    """Command-line interface entry point for the evaluation system.

    Parses command-line arguments, loads environment variables from .env,
    and dispatches to the main() function with the selected evaluation
    configuration. Available evaluation names are defined in config.py.

    Command-line Arguments:
        --eval_name: Name of the evaluation configuration to run.
            Must match an eval_name defined in config.py. Defaults to "test".

    Raises:
        ValueError: If the specified eval_name is not found in config.py
            or if multiple configurations share the same eval_name.

    Example:
        Run from command line::

            python -m src.main --eval_name production
    """
    load_dotenv()

    # Check for critical API key
    if os.getenv("VECTARA_API_KEY") is None:
        logger.critical("VECTARA_API_KEY not found in environment variables. Judging will fail.")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Exiting due to missing VECTARA_API_KEY.")
            return

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

    # Check for missing API keys based on active configs
    required_keys = set()
    for llm_config in eval_config.per_LLM_configs:
        api_type = getattr(llm_config, 'api_type', 'default')
        if api_type == "huggingface":
            continue  # Uses cached login, no API key needed
        elif api_type == "default":
            company = getattr(llm_config, 'company', '')
            if company:
                # Replace hyphens with underscores for valid env var names
                key_name = company.upper().replace("-", "_")
                required_keys.add(f"{key_name}_API_KEY")
        else:
            key_name = api_type.upper().replace("-", "_")
            required_keys.add(f"{key_name}_API_KEY")

    missing_keys = [key for key in required_keys if os.getenv(key) is None]
    if missing_keys:
        for key in missing_keys:
            logger.warning(f"API key not found: {key}")
        response = input("Some API keys are missing. Continue anyway? Program will crash if these keys are needed. (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Exiting due to missing API keys.")
            return

    main(eval_config)

if __name__ == "__main__":
    cli_main()