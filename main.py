from src.logging.Logger import logger
from src.scripts import get_summaries, get_hhem_scores
from dotenv import load_dotenv

from src.LLMs.OpenAI_GPTd4p1.GPTd4p1 import GPTd4p1



if __name__ == "__main__":
    load_dotenv()
    models = [GPTd4p1()]
    logger.log("Starting main program")
    get_summaries.run(models)
    # get_hhem_scores.run(models)
    logger.log("Main program exiting.")