from src.tests.AbstractTest import AbstractTest
from src.LLMs.AbstractLLM import AbstractLLM
from src.config import TEST_DATA_PATH
from src.utils.json_utils import load_json, json_exists
from src.utils.build_utils import builds_models, process_raw_config
from src.logging.Logger import logger
import csv

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
        self.config_path = "test_config.json"

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
        valid_config = None
        if json_exists(self.config_path):
            raw_config = load_json(self.config_path)
            valid_config = process_raw_config(raw_config)
        else:
            logger.log(f"{self.config_path} not found, exiting")
            print(
                f"WARNING: {self.config_path} not found, cannot perform LLM "
                "tests without it."
            )
            return

        models = builds_models(valid_config)

        logger.log("Testing LLM functionality")
        for model in models:
            logger.log(f"Running tests on {model.get_model_name()}")
            self.set_model(model)
            self.test_summarize(self.sample_article)
            logger.log(f"{model.get_model_name()} passed")
        logger.log("Finished testing models")

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