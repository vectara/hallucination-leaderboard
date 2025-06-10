from src.logging.Logger import logger
from src.tests.TestLLM import TestLLM
from dotenv import load_dotenv


from src.utils.json_utils import json_exists, load_json
from src.utils.build_utils import builds_models
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

    config = None
    if json_exists("config_test.json"):
        config = load_json("config_test.json")
    else:
        logger.log("config_test.json not found, exiting")
        return

    models = builds_models(config)

    logger.log("Testing LLM functionality")
    for model in models:
        logger.log(f"Running tests on {model.get_model_name()}")
        llm_tester.set_model(model)
        llm_tester.run_tests()
        logger.log(f"{model.get_model_name()} passed")
    logger.log("Finished testing models")

if __name__ == "__main__":
    load_dotenv()
    run_tests()