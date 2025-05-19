from src.logging.Logger import logger
from src.tests.TestLLM import TestLLM

from src.LLMs.AbstractLLM import AbstractLLM
from src.LLMs.OpenAI_GPTd4p1.GPTd4p1 import GPTd4p1

def run_tests():
    logger.log("Running Tests...")
    test_models()
    logger.log("Tests Completed")

def test_models():
    logger.log("Testing models")
    llm_tester = TestLLM()
    models = [GPTd4p1()]
    test_model_summarize(models, llm_tester)
    logger.log("Finished testing models")


def test_model_summarize(models: list[AbstractLLM], llm_tester: TestLLM):
    logger.log("Testing LLM summarize functionality")
    for model in models:
        logger.log(f"Running tests on {model.get_name()}")
        llm_tester.set_model(model)
        llm_tester.run_tests()
        logger.log(f"{model.get_name()} passed")

if __name__ == "__main__":
    run_tests()