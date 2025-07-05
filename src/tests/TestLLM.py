from . AbstractTest import AbstractTest
from .. LLMs.AbstractLLM import AbstractLLM
from .. data_model import EvalConfig
from .. config import CONFIG
from .. LLMs.AbstractLLM import build_models
from .. Logger import logger
import csv

# Test data file path
TEST_DATA_PATH = "datasets/test_data.csv"

class TestLLM(AbstractTest):
    """
    Test that the given LLMs summarize method works

    Attributes:
        sample_article (str): Example article for testing
        model (AbstractLLM): model being tested
        config_path (str): model setup config file

    Methods:
        test_summarize(test_article)
        set_model(model)
    """
    def __init__(self):
        super().__init__()
        self.sample_article = None
        with open(TEST_DATA_PATH) as csvfile:
            reader = csv.DictReader(csvfile)
            first_row = next(reader)
            self.sample_article = first_row["text"]
        self.model = None

    def __str__(self):
        return "TestLLM"

    def run_tests(self):
        """
        Tests critical functionality of LLMs. Currently this is just the ability
        to summarize a document

        Args:
            None
        
        Returns:
            None
        """
        config = EvalConfig(**CONFIG)
        models = build_models(config.LLMs_to_eval)
        logger.info("Testing LLM functionality")
        for model in models:
            logger.info(f"Running tests on {model.get_model_name()}")
            self.set_model(model)
            self.test_summarize(self.sample_article)
            logger.info(f"{model.get_model_name()} passed")
        logger.info("Finished testing models")

    def test_summarize(self, test_article: str):
        """
        Tests to see if a model can take in a string and output a string.

        Args:
            test_article (str): Article to be summarized by model
        
        Returns:
            None
        """

        with self.model as m: 
            msg = self.model.summarize(test_article)
        assert type(msg) == str

    def set_model(self, model: AbstractLLM):
        """
        Sets current model

        Args:
            model (AbstractLLM): model to be tested

        Returns:
            None
        """
        self.model = model