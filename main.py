from src.logging.Logger import logger
from src.scripts import get_summaries, get_hhem_scores, combine_hhem_scores, get_results
from dotenv import load_dotenv
import pandas as pd
import argparse
import os

from src.LLMs.OpenAI.GPTd4p1 import GPTd4p1
from src.LLMs.Fanar import Fanar
from src.LLMs.DeepSeekAI import DeepSeekAI
from src.LLMs.Anthropic.ClaudeOpus4p0 import ClaudeOpus4p0
from src.LLMs.Anthropic.ClaudeSonnet4p0 import ClaudeSonnet4p0

#TODO: Fix metrics answer rate, new class system revamp


def main(args: argparse.ArgumentParser):
    data_path = None
    if args.test:
        print("Using test data, if this is not a test run results are not useful")
        data_path = os.getenv("TEST_DATA")
    else:
        data_path = os.getenv("LB_DATA")
    
    models = [GPTd4p1()]
    # models = [GPTd4p1(), ClaudeSonnet4p0(), ClaudeOpus4p0()]
    # models = [ClaudeSonnet4p0(), ClaudeOpus4p0()]
    # models = [Fanar("Fanar")]
    # models = [DeepSeekAI("DeepSeek-R1-0528")]

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
        combine_hhem_scores.run(models)
        get_results.run(models)

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