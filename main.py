from src.logging.Logger import logger
from src.scripts import get_summaries, get_judgements, combine_hhem_scores, get_results
from dotenv import load_dotenv
import pandas as pd
import argparse
import os
from src.utils.json_utils import load_json, json_exists
from src.LLMs.model_registry import MODEL_REGISTRY
from src.utils.build_utils import builds_models
from src.config import TEST_DATA_PATH, LB_DATA_PATH
import src.LLMs


#TODO: Standard Dev

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

    config = None
    if json_exists("config.json"):
        config = load_json("config.json")
    else:
        logger.log("No Config file was found, exiting")
        return

    models = builds_models(config)

    if args.process == "get_summ":
        article_df = pd.read_csv(data_path)
        get_summaries.run(models, article_df, force=args.force)
    elif args.process == "get_judge":
        article_df = pd.read_csv(data_path)
        get_judgements.run(models, article_df, force=args.force)
    elif args.process == "combine_hhem":
        print("Combine hhem is removed for now since outdated")
        pass
        # combine_hhem_scores.run(models)
    elif args.process == "get_results":
        get_results.run(models)
    elif args.process == "get_summ_judge":
        article_df = pd.read_csv(data_path)
        get_summaries.run(models, article_df, force=args.force)
        get_judgements.run(models, article_df, force=args.force)
    elif args.process == "get_summ_judge_results":
        article_df = pd.read_csv(data_path)
        get_summaries.run(models, article_df, force=args.force)
        get_judgements.run(models, article_df, force=args.force)
        get_results.run(models)
    else:
        print("No program type was specified, exiting program. Run program with --help flag for info")
    # else:
    #     get_summaries.run(models, force=args.force)
    #     get_hhem_scores.run(models, force=args.force)
    #     # combine_hhem_scores.run(models)
    #     get_results.run(models)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="HHEM Leaderboard Backend",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "process",
        choices=["get_summ", "get_judge", "combine_hhem", "get_results", "get_summ_judge", "get_summ_judge_results"],
        nargs="?",
        help=(
            "Run a specific process. All will run if not specified.\n"
            "   get_summ      - generates and stores summaries for all models "
            "in a JSON file\n"
            "   get_judge      - generates and stores metrics corresponding llm summaries "
            "models in a JSON file\n"
            "   combine_hhem  - combines HHEM scores for all models into a "
            "singular JSON file\n"
            "   get_results   - computers final metrics for display on LB\n"
            "   get_summ_judge - performs get_summ > get_judge\n"
            "   get_summ_judge_results - performs get_summ > get_judge > get_results\n"
        )
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Loads test data instead of Leaderboard Data"
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Forces get_summary and/or get_hhem process to regenerate all JSON "
            "files even if they exist"
        )
    )

    args = parser.parse_args()
    logger.log("Starting main program")
    main(args)
    logger.log("Main program exiting.")