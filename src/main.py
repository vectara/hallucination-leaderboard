from typing import List

from dotenv import load_dotenv
import pandas as pd
import argparse

from . data_model import EvalConfig, ModelInstantiationError, SourceArticle
from . LLMs import AbstractLLM, MODEL_REGISTRY
from . Logger import logger
from . pipeline import (
    get_summaries
    # get_summaries, get_judgements, get_results
)

def main(eval_config: EvalConfig):
    """
    Main function for program

    Args:
        args (argparse.ArgumentParser): console arguments
    """

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

    # TODO: Add judge and reduce pipelines
    # if "judge" in eval_config.pipeline:
    #     get_judgements.run(models, article_df)

    # if "reduce" in eval_config.pipeline:
    #     get_results.run(models)

    # if args.process == GET_SUMM:
    #     get_summaries.run(models, article_df, ow=args.overwrite)
    # elif args.process == GET_JUDGE:
    #     get_judgements.run(models, article_df)
    # elif args.process == GET_RESULTS:
    #     get_results.run(models)
    # elif args.process == "get_summ_judge":
    #     get_summaries.run(models, article_df, ow=args.overwrite)
    #     get_judgements.run(models, article_df)
    # elif args.process == "get_judge_results":
    #     get_judgements.run(models, article_df)
    #     get_results.run(models)
    # elif args.process == "get_summ_judge_results":
    #     get_summaries.run(models, article_df, ow=args.overwrite)
    #     get_judgements.run(models, article_df)
    #     get_results.run(models)
    # else:
    #     print(
    #         "No process was specified, running instructions specified in "
    #         "config.py instead. Run program with --help flag for info"
    #     )
    #     config_run(config, models)

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
#         get_judgements.run(models, article_df)
#     elif config.pipeline == [GET_RESULTS]:
#         get_results.run(models)
#     elif config.pipeline == [GET_SUMM, GET_JUDGE]:
#         get_summaries.run(models, article_df, ow=config.overwrite)
#         get_judgements.run(models, article_df)
#     elif config.pipeline == [GET_JUDGE, GET_RESULTS]:
#         get_judgements.run(models, article_df)
#         get_results.run(models)
#     elif config.pipeline == [GET_SUMM, GET_JUDGE, GET_RESULTS]:
#         get_summaries.run(models, article_df, ow=config.overwrite)
#         get_judgements.run(models, article_df)
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
        "--config_key",
        choices=[c["eval_name"] for c in eval_configs],
        help="Which evaluation to run. For test, select 'test'.",
        default="test"
    )

    args = parser.parse_args()
    config_key = args.config_key

    # scan eval_configs for the config with the given key
    qualifying_configs = [c for c in eval_configs if c["eval_name"] == config_key]
    if len(qualifying_configs) == 0:
        raise ValueError(f"Evaluation config with key {config_key} not found")
    elif len(qualifying_configs) > 1:
        raise ValueError(f"Multiple evaluation configs with key {config_key} found. Please check config.py. ")
    
    config = EvalConfig(**qualifying_configs[0])

    main(config)

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
#             "                            get_summ_judge to ensure judgements\n"
#             "                            are synchronized\n"
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
