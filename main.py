from src.logging.Logger import logger
from src.scripts import get_summaries, get_hhem_scores, combine_hhem_scores, get_results
from dotenv import load_dotenv
import pandas as pd
import argparse
import os
from src.utils.json_utils import load_json, json_exists
from src.LLMs.model_registry import MODEL_REGISTRY
import src.LLMs


#TODO: LLM Class Revamp, Documentation, Standard Dev

def main(args: argparse.ArgumentParser):
    data_path = None
    if args.test:
        print("Using test data, if this is not a test run results are not useful")
        data_path = os.getenv("TEST_DATA")
    else:
        data_path = os.getenv("LB_DATA")

    config = None
    if json_exists("config.json"):
        config = load_json("config.json")
    else:
        logger.log("No Config file was found, exiting")
        return

    models = builds_models(config)
    
    # models = [GPTd4p1(), ClaudeSonnet4p0(), ClaudeOpus4p0()]

    if args.process == "get_summ":
        article_df = pd.read_csv(data_path)
        get_summaries.run(models, article_df, force=args.force)
    elif args.process == "get_hhem":
        article_df = pd.read_csv(data_path)
        get_hhem_scores.run(models, article_df, force=args.force)
    elif args.process == "combine_hhem":
        combine_hhem_scores.run(models)
    elif args.process == "get_results":
        get_results.run(models)
    elif args.process == "get_summ_hhem":
        article_df = pd.read_csv(data_path)
        get_summaries.run(models, article_df, force=args.force)
        get_hhem_scores.run(models, article_df, force=args.force)
    elif args.process == "get_summ_hhem_results":
        article_df = pd.read_csv(data_path)
        get_summaries.run(models, article_df, force=args.force)
        get_hhem_scores.run(models, article_df, force=args.force)
        get_results.run(models)
    else:
        get_summaries.run(models, force=args.force)
        get_hhem_scores.run(models, force=args.force)
        # combine_hhem_scores.run(models)
        get_results.run(models)

def builds_models(config):
    models = []
    for entry in config:
        company = entry.get("company")
        params = entry.get("params", {})

        if not company:
            logger.log("Missing Company key, skipping")
            continue

        model_class = MODEL_REGISTRY.get(company)
        if not model_class:
            logger.log("No registered model for this company, skipping")
            continue
            
        print(f"adding {company}")

        try:
            models.append(model_class(**params))
        except Exception as e:
            logger.log(f"Failed to instantiate {company} model: {e}")

    return models

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="HHEM Leaderboard Backend",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "process",
        choices=["get_summ", "get_hhem", "combine_hhem", "get_results", "get_summ_hhem", "get_summ_hhem_results"],
        nargs="?",
        help=(
            "Run a specific process. All will run if not specified.\n"
            "   get_summ      - generates and stores summaries for all models "
            "in a JSON file\n"
            "   get_hhem      - generates and stores HHEM scores for all "
            "models in a JSON file\n"
            "   combine_hhem  - combines HHEM scores for all models into a "
            "singular JSON file\n"
            "   get_results   - computers final metrics for display on LB\n"
            "   get_summ_hhem - performs get_summ then get_hhem\n"
            "   get_summ_hhem_results - performs get_summ > get_hhem > get_results\n"
            "If none specified all will run: (get_summ>get_hhem>combine_hhem)"
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