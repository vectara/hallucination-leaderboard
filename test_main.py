from src.logging.Logger import logger
from src.tests.TestLLM import TestLLM
from dotenv import load_dotenv

from src.LLMs.AbstractLLM import AbstractLLM
from src.LLMs.OpenAI_GPTd4p1.GPTd4p1 import GPTd4p1

"""
Runs program critical tests for functionality 

Functions:
    run_tests()
    test_models()
"""

def run_tests():
    """
    Main function where all tests are called

    Args:
        None
    Returns:
        None
    """
    logger.log("Running Tests...")
    test_models()
    logger.log("Tests Completed")

def test_models():
    """
    Tests model objects for critical functionality. New objects need to be added
    here as they are added to the program.

    Args:
        None
    Returns:
        None
    """
    logger.log("Testing models")
    llm_tester = TestLLM()

    '''Add new models in list below'''
    models = [GPTd4p1()]

    logger.log("Testing LLM functionality")
    for model in models:
        logger.log(f"Running tests on {model.get_name()}")
        llm_tester.set_model(model)
        llm_tester.run_tests()
        logger.log(f"{model.get_name()} passed")
    logger.log("Finished testing models")

if __name__ == "__main__":
    load_dotenv()
    run_tests()