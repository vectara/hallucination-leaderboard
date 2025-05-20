from src.logging.Logger import logger
from src.scripts import get_summaries
from dotenv import load_dotenv



if __name__ == "__main__":
    load_dotenv()
    logger.log("Starting main program")
    get_summaries.run()
    logger.log("Main program exiting.")