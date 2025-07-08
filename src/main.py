import argparse
from typing import List

import pandas as pd
from dotenv import load_dotenv

from . data_model import EvalConfig, ModelInstantiationError, SourceArticle
from . LLMs import AbstractLLM, MODEL_REGISTRY
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
        raise ValueError(f"Data validation failed: {e}")

    if "summarize" in eval_config.pipeline:
        get_summaries(eval_config, article_df)

    if "judge" in eval_config.pipeline:
        get_judgments(eval_config, article_df)
    
    if "aggregate" in eval_config.pipeline:
        aggregate_judgments(eval_config)


# Disabled as now we solely use configuration file to sepecify the pipeline. -- Forrest, 2025-07-03
# def config_run(config: Config, models: list[AbstractLLM]):
#     article_df = pd.read_csv(config.source_article_path)
#     if config.overwrite:
#         confirmation = input(
#             "\nOverwrite is enabled in the given config. "
#             "Are you sure you want to overwrite? [y/N]: "
#         )
#         if confirmation.lower() not in ("y", "yes"):
#             print("Aborting Run")
#             exit(0)

#     if config.pipeline == [GET_SUMM]:
#         get_summaries.run(models, article_df, ow=config.overwrite)
#     elif config.pipeline == [GET_JUDGE]:
#         get_judgments.run(models, article_df)
#     elif config.pipeline == [GET_RESULTS]:
#         get_results.run(models)
#     elif config.pipeline == [GET_SUMM, GET_JUDGE]:
#         get_summaries.run(models, article_df, ow=config.overwrite)
#         get_judgments.run(models, article_df)
#     elif config.pipeline == [GET_JUDGE, GET_RESULTS]:
#         get_judgments.run(models, article_df)
#         get_results.run(models)
#     elif config.pipeline == [GET_SUMM, GET_JUDGE, GET_RESULTS]:
#         get_summaries.run(models, article_df, ow=config.overwrite)
#         get_judgments.run(models, article_df)
#         get_results.run(models)

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


# Disabled because we use config file to specify the pipeline. -- Forrest, 2025-07-03
# if __name__ == "__main__":
#     #TODO: Rethink how to name pipelines
#     load_dotenv()
#     parser = argparse.ArgumentParser(
#         description="HHEM Leaderboard Backend",
#         formatter_class=argparse.RawTextHelpFormatter
#     )
#     parser.add_argument(
#         "process",
#         choices=[
#             GET_SUMM,
#             GET_JUDGE,
#             GET_RESULTS, 
#             "get_summ_judge",
#             "get_judge_results",
#             "get_summ_judge_results",

#         ],
#         nargs="?",
#         help=(
#             "Run a specific process.\n"
#             f"   {GET_SUMM}              - generate and save summaries from\n"
#             "                            config enabled models in jsonl file.\n"
#             "                            It's best to avoid get_summ and run\n"
#             "                            get_summ_judge to ensure judgments\n"
#             f"   {GET_JUDGE}              - compute and save metrics for the\n"
#             "                            generated summaries in jsonl file\n"
#             f"   {GET_RESULTS}            - compute and save aggregate stats\n"
#             "                            for the computed metrics in json file\n"
#             "   get_summ_judge         - performs get_summ > get_judge\n"
#             "   get_judge_results      - performs get_judge > get_results\n"
#             "   get_summ_judge_results - performs get_summ > get_judge >\n"
#             "                            get_results\n"
#         )
#     )

#     parser.add_argument(
#         "--test",
#         action="store_true",
#         help=(
#             "Loads test data instead of leaderboard data"
#         )
#     )

#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#         help=(
#             "Forces get_summ process to overwrite jsonl if it exists"
#         )
#     )

#     args = parser.parse_args()
#     logger.info("Starting main program")
#     main(args)
#     logger.info("Main program exiting.")
