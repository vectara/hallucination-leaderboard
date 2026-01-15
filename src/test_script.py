"""Test runner for critical functionality verification.

This module provides the entry point for running the HHEM Leaderboard
test suite. Executes tests for analytics functions and LLM integrations
to verify core pipeline functionality before deployment.

Functions:
    run_tests: Main entry point that orchestrates all test execution.
    test_models: Iterator that runs each test class in sequence.

Example:
    Run from command line::

        python -m src.test_script

    Or programmatically::

        from src.test_script import run_tests
        run_tests()
"""

from dotenv import load_dotenv

from . Logger import logger
from . tests.AbstractTest import AbstractTest
from . tests.TestLLM import TestLLM
from . tests.TestAnalytics import TestAnalytics

def run_tests():
    """Execute the complete test suite for the HHEM Leaderboard system.

    Main entry point that instantiates and runs all registered test classes.
    Currently executes TestAnalytics and TestLLM tests in sequence, logging
    progress to both console and file.

    Note:
        New test classes should be added to the tests list as they are
        created. All test classes must inherit from AbstractTest.
    """
    logger.info("Running Tests...")
    tests = [TestAnalytics(), TestLLM()]
    test_models(tests)
    logger.info("Tests Completed")

def test_models(tests: list[AbstractTest]):
    """Iterate through and execute a list of test classes.

    Runs the run_tests() method on each provided test instance, logging
    the start and completion of each test class for traceability.

    Args:
        tests: List of AbstractTest subclass instances to execute.
            Each test class must implement the run_tests() method.
    """
    for test in tests:
        logger.info(f"Running {test.__str__()}")
        test.run_tests()
        logger.info(f"Finished Running {test.__str__()}")

if __name__ == "__main__":
    load_dotenv()
    run_tests()