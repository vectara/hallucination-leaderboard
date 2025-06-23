from src.logging.Logger import logger
from src.scripts import (
    get_summaries, get_judgements, get_results
)
from dotenv import load_dotenv
import pandas as pd
import argparse
from src.utils.json_utils import load_json, file_exists
from src.utils.build_utils import builds_models, process_raw_config
from src.config import (
    TEST_DATA_PATH, LB_DATA_PATH, CONFIG, GET_SUMM, GET_JUDGE, GET_RESULTS
)
from src.data_struct.config_model import Config

"""
Main Program File

Functions:
    main(args)
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

    # valid_model_configs = None
    # if file_exists("config.json"):
    #     raw_model_configs = load_json("config.json")
    #     valid_model_configs = process_raw_config(raw_model_configs)
    # else:
    #     logger.log("No Config file was found, exiting")
    #     return

    # models = builds_models(valid_model_configs)
    models = config.LLMs_to_eval
    article_df = pd.read_csv(data_path)

    if args.process == "get_summ":
        get_summaries.run(models, article_df, ow=args.overwrite)
    elif args.process == "get_judge":
        get_judgements.run(models, article_df)
    elif args.process == "get_results":
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
            (
                "No process was specified, running instructions specified in "
                "config.py instead. Run program with --help flag for info"
            )
        )

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
    logger.log("Starting main program")
    main(args)
    logger.log("Main program exiting.")
