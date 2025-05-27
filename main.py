from src.logging.Logger import logger
from src.scripts import get_summaries, get_hhem_scores, combine_hhem_scores
from dotenv import load_dotenv
import argparse

from src.LLMs.OpenAI_GPTd4p1.GPTd4p1 import GPTd4p1
from src.LLMs.Anthropic.Anthropic_ClaudeOpus4p0.ClaudeOpus4p0 import ClaudeOpus4p0
from src.LLMs.Anthropic.Anthropic_ClaudeSonnet4p0.ClaudeSonnet4p0 import ClaudeSonnet4p0


def main(args: argparse.ArgumentParser):
    # models = [GPTd4p1(), ClaudeSonnet4p0(), ClaudeOpus4p0()]
    # models = [ClaudeSonnet4p0(), ClaudeOpus4p0()]
    models = [GPTd4p1()]

    if args.process == "get_summ":
        get_summaries.run(models, force=args.force)
    elif args.process == "get_hhem":
        get_hhem_scores.run(models, force=args.force)
    elif args.process == "combine_hhem":
        combine_hhem_scores.run(models)
    else:
        get_summaries.run(models, force=args.force)
        get_hhem_scores.run(models, force=args.force)
        combine_hhem_scores.run(models)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="HHEM Leaderboard Backend",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "process",
        choices=["get_summ", "get_hhem", "combine_hhem"],
        nargs="?",
        help=(
            "Run a specific process. All will run if not specified.\n"
            "   get_summ      - generates and stores summaries for all models "
            "in a JSON file\n"
            "   get_hhem      - generates and stores HHEM scores for all "
            "models in a JSON file\n"
            "   combine_hhem  - combines HHEM scores for all models into a "
            "singular JSON file\n"
            "If none specified all will run: (get_summ>get_hhem>combine_hhem)"
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