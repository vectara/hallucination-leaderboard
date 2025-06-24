from src.logging.Logger import logger
from src.tests.AbstractTest import AbstractTest
from src.tests.TestLLM import TestLLM
from src.tests.TestAnalytics import TestAnalytics
from dotenv import load_dotenv
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
    logger.info("Running Tests...")
    tests = [TestAnalytics(), TestLLM()]
    test_models(tests)
    logger.info("Tests Completed")

def test_models(tests: list[AbstractTest]):
    """
    Tests model objects for critical functionality. New objects need to be added
    here as they are added to the program.

    Args:
        None
    Returns:
        None
    """
    for test in tests:
        logger.info(f"Running {test.__str__()}")
        test.run_tests()
        logger.info(f"Finished Running {test.__str__()}")

if __name__ == "__main__":
    load_dotenv()
    run_tests()