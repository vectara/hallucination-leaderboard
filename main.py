from src.Logger import logger
from src.scripts import (
    get_summaries, get_judgements, get_results
)
from dotenv import load_dotenv
import pandas as pd
import argparse
from src.constants import (
    TEST_DATA_PATH, LB_DATA_PATH, GET_SUMM, GET_JUDGE, GET_RESULTS
)
from src.config import CONFIG   
from src.data_struct.config_model import Config, ModelConfig
from src.LLMs.AbstractLLM import AbstractLLM, MODEL_REGISTRY

"""
Main Program File

Functions:
    main(args)
    build_models(llm_configs)
"""

def main(args: argparse.ArgumentParser):
    """
    Main function for program

    Args:
        args (argparse.ArgumentParser): console arguments
    """

    data_path = None
    if args.test:
        print("Using test data, if this is not a test run results are not useful")
        data_path = TEST_DATA_PATH
    else:
        data_path = LB_DATA_PATH

    config = Config(**CONFIG)

    models = build_models(config.LLMs_to_eval)
    article_df = pd.read_csv(data_path)

    if args.process == GET_SUMM:
        get_summaries.run(models, article_df, ow=args.overwrite)
    elif args.process == GET_JUDGE:
        get_judgements.run(models, article_df)
    elif args.process == GET_RESULTS:
        get_results.run(models)
    elif args.process == "get_summ_judge":
        get_summaries.run(models, article_df, ow=args.overwrite)
        get_judgements.run(models, article_df)
    elif args.process == "get_judge_results":
        get_judgements.run(models, article_df)
        get_results.run(models)
    elif args.process == "get_summ_judge_results":
        get_summaries.run(models, article_df, ow=args.overwrite)
        get_judgements.run(models, article_df)
        get_results.run(models)
    else:
        print(
            "No process was specified, running instructions specified in "
            "config.py instead. Run program with --help flag for info"
        )
        config_run(config, models)


def config_run(config: Config, models: list[AbstractLLM]):
    article_df = pd.read_csv(config.input_file)
    if config.overwrite:
        confirmation = input(
            "\nOverwrite is enabled in the given config. "
            "Are you sure you want to overwrite? [y/N]: "
        )
        if confirmation.lower() not in ("y", "yes"):
            print("Aborting Run")
            exit(0)

    if config.pipeline == [GET_SUMM]:
        get_summaries.run(models, article_df, ow=config.overwrite)
    elif config.pipeline == [GET_JUDGE]:
        get_judgements.run(models, article_df)
    elif config.pipeline == [GET_RESULTS]:
        get_results.run(models)
    elif config.pipeline == [GET_SUMM, GET_JUDGE]:
        get_summaries.run(models, article_df, ow=config.overwrite)
        get_judgements.run(models, article_df)
    elif config.pipeline == [GET_JUDGE, GET_RESULTS]:
        get_judgements.run(models, article_df)
        get_results.run(models)
    elif config.pipeline == [GET_SUMM, GET_JUDGE, GET_RESULTS]:
        get_summaries.run(models, article_df, ow=config.overwrite)
        get_judgements.run(models, article_df)
        get_results.run(models)

def build_models(llm_configs: list[ModelConfig]) -> list[AbstractLLM]:
    """
    Builds the models given in the config list if it is registered

    Args:
        config (list[dict]): list of dictionaries for model object init

    Returns:
        list[AbstractLLM]: list of models
    """

    models = []
    for model in llm_configs:
        company_class = MODEL_REGISTRY.get(model.company)
        if company_class == None:
            logger.warning("No registered class for this company, skipping")
            print(f"This {company_class} is not registered, can't build")
            continue

        try:
            models.append(company_class(**model.params.model_dump()))
        except Exception as e:
            logger.warning(
                f"failed to instantiate {model.company}-"
                f"{model.params.model_name}-{model.params.date_code} : {e}"
            )
            print(
                f"failed to instantiate {model.company}-"
                f"{model.params.model_name}-{model.params.date_code} : {e}"
            )
    return models

if __name__ == "__main__":
    #TODO: Rethink how to name pipelines
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="HHEM Leaderboard Backend",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "process",
        choices=[
            GET_SUMM,
            GET_JUDGE,
            GET_RESULTS, 
            "get_summ_judge",
            "get_judge_results",
            "get_summ_judge_results",

        ],
        nargs="?",
        help=(
            "Run a specific process.\n"
            f"   {GET_SUMM}              - generate and save summaries from\n"
            "                            config enabled models in jsonl file.\n"
            "                            It's best to avoid get_summ and run\n"
            "                            get_summ_judge to ensure judgements\n"
            "                            are synchronized\n"
            f"   {GET_JUDGE}              - compute and save metrics for the\n"
            "                            generated summaries in jsonl file\n"
            f"   {GET_RESULTS}            - compute and save aggregate stats\n"
            "                            for the computed metrics in json file\n"
            "   get_summ_judge         - performs get_summ > get_judge\n"
            "   get_judge_results      - performs get_judge > get_results\n"
            "   get_summ_judge_results - performs get_summ > get_judge >\n"
            "                            get_results\n"
        )
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Loads test data instead of leaderboard data"
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Forces get_summ process to overwrite jsonl if it exists"
        )
    )

    args = parser.parse_args()
    logger.info("Starting main program")
    main(args)
    logger.info("Main program exiting.")
